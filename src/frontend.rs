//! Your exercise: lower a supported Python subset to native MLIR via Melior.
use anyhow::{bail, ensure, Context as _, Result};
use melior::{
    dialect::{arith, func},
    ir::{
        self,
        attribute::{DenseI32ArrayAttribute, FloatAttribute, StringAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, IntegerType, RankedTensorType},
        Attribute, BlockLike, BlockRef, Identifier, Module, Operation, Region, RegionLike, Type,
        Value, ValueLike,
    },
    Context,
};
use rumpy::{DType, PyNdArray};
use rustpython_parser::{
    ast::{self},
    Parse,
};
use rustpython_vm::{
    builtins::{PyFloat, PyInt},
    AsObject, PyObjectRef, VirtualMachine,
};
use std::collections::HashMap;

/// Parse Python directly in Rust without executing it.
pub fn parse_source(source: &str, filename: &str) -> Result<ast::Suite> {
    ast::Suite::parse(source, filename).map_err(|error| anyhow::anyhow!("{filename}: {error}"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Typ {
    Unknown,
    /// Python's arbitrary-precision int; not an implicit MLIR i64.
    Integral,
    /// Python's double-precision float.
    Fractional,
    Boolean,
    Tensor {
        dtype: DType,
        shape: Vec<usize>,
    },
}

impl Typ {
    fn elementwise_result(&self, op: &str, rhs: &Self) -> Result<Self> {
        match (self, rhs) {
            (
                Self::Tensor {
                    dtype: left_dtype,
                    shape: left,
                },
                Self::Tensor {
                    dtype: right_dtype,
                    shape: right,
                },
            ) => {
                ensure!(left_dtype == right_dtype,
                    "Elementwise {op} requires matching dtypes, got {left_dtype:?} and {right_dtype:?}");
                let rank = left.len().max(right.len());
                let mut shape = vec![1; rank];
                for offset in 0..rank {
                    let l = left.iter().rev().nth(offset).copied().unwrap_or(1);
                    let r = right.iter().rev().nth(offset).copied().unwrap_or(1);
                    ensure!(
                        l == r || l == 1 || r == 1,
                        "Cannot broadcast shapes {left:?} and {right:?} for {op}"
                    );
                    shape[rank - 1 - offset] = if l == 1 { r } else { l };
                }
                Ok(Self::Tensor {
                    dtype: *left_dtype,
                    shape,
                })
            }
            // Python float scalars take the array's element type in the
            // supported float32 tensor subset; standalone floats remain f64.
            (
                tensor @ Self::Tensor {
                    dtype: DType::F32, ..
                },
                Self::Fractional,
            )
            | (
                Self::Fractional,
                tensor @ Self::Tensor {
                    dtype: DType::F32, ..
                },
            ) => Ok(tensor.clone()),
            (Self::Integral, Self::Integral) | (Self::Fractional, Self::Fractional) => {
                Ok(self.clone())
            }
            _ => bail!(
                "Elementwise {op} requires compatible operand types, got {self:?} and {rhs:?}"
            ),
        }
    }

    /// Check the supported binary operations and infer their result type.
    /// Matrix multiplication supports rank-2 tensors. Elementwise operations
    /// broadcast trailing dimensions without promoting tensor dtypes.
    fn binary_result(&self, op: ast::Operator, rhs: &Self) -> Result<Self> {
        match op {
            ast::Operator::MatMult => {
                let (
                    Self::Tensor {
                        dtype: left_dtype,
                        shape: left,
                    },
                    Self::Tensor {
                        dtype: right_dtype,
                        shape: right,
                    },
                ) = (self, rhs)
                else {
                    bail!("Matrix multiplication requires two matrices, got {self:?} and {rhs:?}");
                };
                ensure!(
                    left.len() == 2 && right.len() == 2,
                    "Matrix multiplication requires rank-2 tensors, got shapes {left:?} and {right:?}"
                );
                ensure!(
                    left_dtype == right_dtype,
                    "Matrix multiplication requires matching dtypes, got {left_dtype:?} and {right_dtype:?}"
                );
                ensure!(
                    left[1] == right[0],
                    "Matrix multiplication inner dimensions must match: {left:?} @ {right:?} ({} != {})",
                    left[1], right[0]
                );
                Ok(Self::Tensor {
                    dtype: *left_dtype,
                    shape: vec![left[0], right[1]],
                })
            }
            ast::Operator::Add | ast::Operator::Mult => {
                self.elementwise_result(&format!("{op:?}"), rhs)
            }
            _ => bail!("Binary operator {op:?} is not supported yet"),
        }
    }

    /// Inspect metadata only: no tensor data is copied or function body executed.
    pub fn from_runtime(value: &PyObjectRef, vm: &VirtualMachine) -> Result<Self> {
        if let Some(array) = value.downcast_ref::<PyNdArray>() {
            let view = array.view();
            let dtype = view.dtype();
            ensure!(dtype.is_numeric(), "Unsupported tensor dtype: {dtype:?}");
            return Ok(Self::Tensor {
                dtype,
                shape: view.shape().to_vec(),
            });
        }
        // Python bool is an int subclass, so check it first.
        if value.is(&vm.ctx.true_value) || value.is(&vm.ctx.false_value) {
            return Ok(Self::Boolean);
        }
        if value.downcast_ref::<PyInt>().is_some() {
            return Ok(Self::Integral);
        }
        if value.downcast_ref::<PyFloat>().is_some() {
            return Ok(Self::Fractional);
        }
        bail!(
            "Unsupported runtime argument: expected a Python bool/int/float or numeric Rumpy array"
        )
    }
}

#[derive(Debug)]
pub struct CompileUnit<'c, 'a> {
    function_name: String,
    variables: HashMap<String, (Typ, Option<Value<'c, 'a>>)>,
    // The context and owning regions must outlive these borrowed blocks.
    block_stack: Vec<BlockRef<'c, 'a>>,
    return_typ: Option<Typ>,
}

impl<'c, 'a> CompileUnit<'c, 'a> {
    /// Bind one function's explicit positional arguments, including positional-only
    /// parameters. Defaults must be supplied by the caller; keyword/variadic
    /// argument binding is not implemented yet.
    /// HACK: we don't update variable types, this is fine for now
    /// we need to JIT instead of this static stuff eventually.
    pub fn from_runtime_args(
        function: &ast::StmtFunctionDef,
        arguments: &[PyObjectRef],
        vm: &VirtualMachine,
    ) -> Result<Self> {
        ensure!(
            function.args.vararg.is_none()
                && function.args.kwarg.is_none()
                && function.args.kwonlyargs.is_empty(),
            "{}: keyword-only and variadic parameters are not supported yet",
            function.name,
        );
        let parameters: Vec<_> = function
            .args
            .posonlyargs
            .iter()
            .chain(&function.args.args)
            .collect();
        ensure!(
            parameters.len() == arguments.len(),
            "{}: expected {} explicit positional arguments, got {} (supply defaults explicitly)",
            function.name,
            parameters.len(),
            arguments.len()
        );
        let variables = parameters
            .into_iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                let name = parameter.def.arg.to_string();
                let ty = Typ::from_runtime(value, vm)
                    .with_context(|| format!("{} argument '{name}'", function.name))?;
                Ok((name, (ty, None)))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self {
            function_name: function.name.to_string(),
            variables,
            return_typ: None,
            block_stack: vec![],
        })
    }

    pub fn variable_type(&self, name: &str) -> Option<&Typ> {
        self.variables.get(name).map(|(ty, _)| ty)
    }

    /// The SSA value is absent until this variable has been lowered to MLIR.
    pub fn variable_value(&self, name: &str) -> Option<Value<'c, 'a>> {
        self.variables.get(name).and_then(|(_, value)| *value)
    }

    pub fn variables(&self) -> &HashMap<String, (Typ, Option<Value<'c, 'a>>)> {
        &self.variables
    }

    /// Resolve a Python name while lowering the last block on the stack.
    /// Bindings are function-scoped, so entering a nested block does not hide them.
    pub fn get_variable(&self, name: &str) -> Result<Value<'c, 'a>> {
        self.block_stack.last().context("No active block")?;
        let (_, value) = self
            .variables
            .get(name)
            .with_context(|| format!("Undefined variable '{name}'"))?;
        value.with_context(|| format!("Variable '{name}' has no MLIR value yet"))
    }

    /// Bind an existing SSA value while lowering the last block on the stack.
    /// The value can be a block argument, an operation result, or a dominating
    /// value reused by assignment. Emit operations into the block before calling
    /// this method; a name binding itself does not emit an MLIR operation.
    pub fn set_variable(
        &mut self,
        name: impl Into<String>,
        ty: Typ,
        value: Value<'c, 'a>,
    ) -> Result<()> {
        self.block_stack.last().context("No active block")?;
        self.variables.insert(name.into(), (ty, Some(value)));
        Ok(())
    }

    fn emit_arg(&mut self, context: &'c Context, arg: &ast::ArgWithDefault) -> Result<Type<'c>> {
        let (ty, _value) = self
            .variables
            .get(arg.def.arg.as_str())
            .with_context(|| format!("Missing runtime type for '{}'", arg.def.arg))?;

        let mty = match ty {
            Typ::Integral => IntegerType::new(context, 64).into(),
            Typ::Fractional => Type::float64(context),
            Typ::Tensor {
                dtype: DType::F32,
                shape,
            } => {
                let dimensions = shape
                    .iter()
                    .map(|&dimension| u64::try_from(dimension))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                RankedTensorType::new(&dimensions, Type::float32(context), None).into()
            }
            _ => bail!("Argument type {ty:?} is not supported yet"),
        };
        Ok(mty)
    }

    fn compile_name(
        &mut self,
        _context: &'c Context,
        name: &ast::ExprName,
    ) -> Result<(Typ, Value<'c, 'a>)> {
        let (ty, val) = match self.variables.get(name.id.as_str()) {
            Some(val) => val.clone(),
            None => bail!("undefined variable '{}'", name.id),
        };
        let val = val.with_context(|| format!("Variable '{}' has no MLIR value yet", name.id))?;
        Ok((ty, val))
    }
    fn compile_binop(
        &mut self,
        context: &'c Context,
        expr: &ast::ExprBinOp,
    ) -> Result<(Typ, Value<'c, 'a>)> {
        let lhs = self.compile_expr(context, &expr.left)?;
        let rhs = self.compile_expr(context, &expr.right)?;
        let result_ty = lhs.0.binary_result(expr.op, &rhs.0)?;
        let location = ir::Location::unknown(context);
        let operation = match (&result_ty, expr.op) {
            (Typ::Integral, ast::Operator::Add) => arith::addi(lhs.1, rhs.1, location),
            (Typ::Fractional, ast::Operator::Add) => arith::addf(lhs.1, rhs.1, location),
            (Typ::Integral, ast::Operator::Mult) => arith::muli(lhs.1, rhs.1, location),
            (Typ::Fractional, ast::Operator::Mult) => arith::mulf(lhs.1, rhs.1, location),
            (
                Typ::Tensor {
                    dtype: DType::F32, ..
                },
                ast::Operator::Add,
            ) => self.emit_elementwise(context, &lhs, &rhs, &result_ty, "arith.addf")?,
            (
                Typ::Tensor {
                    dtype: DType::F32, ..
                },
                ast::Operator::Mult,
            ) => self.emit_elementwise(context, &lhs, &rhs, &result_ty, "arith.mulf")?,
            (
                Typ::Tensor {
                    dtype: DType::F32, ..
                },
                ast::Operator::MatMult,
            ) => self.emit_matmul(context, lhs.1, rhs.1, &result_ty)?,
            (ty, op) => bail!("binary operator {op:?} is not supported for {ty:?}"),
        };
        let block = self.block_stack.last().context("No active block")?;
        let result = block.append_operation(operation).result(0)?.into();
        Ok((result_ty, result))
    }

    /// Emit a pointwise operation with NumPy-style trailing-dimension broadcast.
    fn emit_elementwise(
        &self,
        context: &'c Context,
        lhs: &(Typ, Value<'c, 'a>),
        rhs: &(Typ, Value<'c, 'a>),
        result_ty: &Typ,
        scalar_op: &str,
    ) -> Result<Operation<'c>> {
        let Typ::Tensor {
            dtype: DType::F32,
            shape,
        } = result_ty
        else {
            bail!("Tensor elementwise lowering requires a float32 result");
        };
        let location = ir::Location::unknown(context);
        let element_type = Type::float32(context);
        let dimensions = shape
            .iter()
            .map(|&d| u64::try_from(d))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let tensor_type: Type = RankedTensorType::new(&dimensions, element_type, None).into();
        let block = self.block_stack.last().context("No active block")?;
        let empty = block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[tensor_type])
                    .build()?,
            )
            .result(0)?
            .into();
        let dims = (0..shape.len())
            .map(|i| format!("d{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut maps = Vec::new();
        let mut operands = Vec::new();
        for (ty, value) in [lhs, rhs] {
            let (indices, value) = match ty {
                Typ::Tensor {
                    shape: input_shape, ..
                } => {
                    let offset = shape.len() - input_shape.len();
                    let indices = input_shape
                        .iter()
                        .enumerate()
                        .map(|(i, &size)| {
                            if size == 1 {
                                "0".into()
                            } else {
                                format!("d{}", offset + i)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    (indices, *value)
                }
                Typ::Fractional => {
                    let narrowed = block
                        .append_operation(
                            OperationBuilder::new("arith.truncf", location)
                                .add_operands(&[*value])
                                .add_results(&[element_type])
                                .build()?,
                        )
                        .result(0)?
                        .into();
                    (String::new(), narrowed)
                }
                _ => bail!("Unsupported elementwise operand: {ty:?}"),
            };
            maps.push(format!("affine_map<({dims}) -> ({indices})>"));
            operands.push(value);
        }
        maps.push(format!("affine_map<({dims}) -> ({dims})>"));
        operands.push(empty);
        let maps = Attribute::parse(context, &format!("[{}]", maps.join(", ")))
            .context("Cannot construct broadcast indexing maps")?;
        let iterators = Attribute::parse(
            context,
            &format!(
                "[{}]",
                vec!["#linalg.iterator_type<parallel>"; shape.len()].join(", ")
            ),
        )
        .context("Cannot construct parallel iterators")?;
        let region = Region::new();
        let body = region.append_block(ir::Block::new(&[(element_type, location); 3]));
        let value = body
            .append_operation(
                OperationBuilder::new(scalar_op, location)
                    .add_operands(&[body.argument(0)?.into(), body.argument(1)?.into()])
                    .add_results(&[element_type])
                    .build()?,
            )
            .result(0)?
            .into();
        body.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[value])
                .build()?,
        );
        Ok(OperationBuilder::new("linalg.generic", location)
            .add_operands(&operands)
            .add_results(&[tensor_type])
            .add_attributes(&[
                (Identifier::new(context, "indexing_maps"), maps),
                (Identifier::new(context, "iterator_types"), iterators),
                (
                    Identifier::new(context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(context, &[2, 1]).into(),
                ),
            ])
            .add_regions([region])
            .build()?)
    }

    fn emit_matmul(
        &self,
        context: &'c Context,
        lhs: Value<'c, 'a>,
        rhs: Value<'c, 'a>,
        result_ty: &Typ,
    ) -> Result<Operation<'c>> {
        let Typ::Tensor {
            dtype: DType::F32,
            shape,
        } = result_ty
        else {
            bail!("Matrix multiplication lowering requires a float32 tensor result");
        };
        let dimensions = shape
            .iter()
            .map(|&dimension| u64::try_from(dimension))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let element_type = Type::float32(context);
        let tensor_type: Type = RankedTensorType::new(&dimensions, element_type, None).into();
        let location = ir::Location::unknown(context);
        let block = self.block_stack.last().context("No active block")?;
        let empty = block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[tensor_type])
                    .build()?,
            )
            .result(0)?
            .into();
        let zero = block
            .append_operation(arith::constant(
                context,
                FloatAttribute::new(context, element_type, 0.0).into(),
                location,
            ))
            .result(0)?
            .into();

        // Matmul computes C += A * B, so tensor.empty alone is not a valid
        // accumulator. Fill it with zero, including when the inner size is zero.
        let fill_region = Region::new();
        let fill_block = fill_region.append_block(ir::Block::new(&[
            (element_type, location),
            (element_type, location),
        ]));
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[fill_block.argument(0)?.into()])
                .build()?,
        );
        let init = block
            .append_operation(
                OperationBuilder::new("linalg.fill", location)
                    .add_operands(&[zero, empty])
                    .add_results(&[tensor_type])
                    .add_attributes(&[(
                        Identifier::new(context, "operandSegmentSizes"),
                        DenseI32ArrayAttribute::new(context, &[1, 1]).into(),
                    )])
                    .add_regions([fill_region])
                    .build()?,
            )
            .result(0)?
            .into();

        let region = Region::new();
        let body = region.append_block(ir::Block::new(&[(element_type, location); 3]));
        let product = body
            .append_operation(arith::mulf(
                body.argument(0)?.into(),
                body.argument(1)?.into(),
                location,
            ))
            .result(0)?
            .into();
        let sum = body
            .append_operation(arith::addf(body.argument(2)?.into(), product, location))
            .result(0)?
            .into();
        body.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[sum])
                .build()?,
        );
        let indexing_maps = Attribute::parse(context,
            "[affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>]"
        ).context("Cannot construct matmul indexing maps")?;
        Ok(OperationBuilder::new("linalg.matmul", location)
            .add_operands(&[lhs, rhs, init])
            .add_results(&[tensor_type])
            .add_attributes(&[
                (
                    Identifier::new(context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(context, &[2, 1]).into(),
                ),
                (Identifier::new(context, "indexing_maps"), indexing_maps),
            ])
            .add_regions([region])
            .build()?)
    }

    fn compile_maximum(
        &mut self,
        context: &'c Context,
        call: &ast::ExprCall,
    ) -> Result<(Typ, Value<'c, 'a>)> {
        let ast::Expr::Attribute(attribute) = call.func.as_ref() else {
            bail!("Only np.maximum calls are supported");
        };
        ensure!(
            attribute.attr.as_str() == "maximum"
                && matches!(attribute.value.as_ref(), ast::Expr::Name(name) if name.id.as_str() == "np"),
            "Only np.maximum calls are supported"
        );
        ensure!(
            !self.variables.contains_key("np"),
            "np.maximum requires an unshadowed np namespace"
        );
        ensure!(
            call.args.len() == 2 && call.keywords.is_empty(),
            "np.maximum requires exactly two positional arguments and no keywords"
        );
        let lhs = self.compile_expr(context, &call.args[0])?;
        let rhs = self.compile_expr(context, &call.args[1])?;
        let ty = lhs.0.elementwise_result("maximum", &rhs.0)?;
        let location = ir::Location::unknown(context);
        let operation = match &ty {
            Typ::Tensor {
                dtype: DType::F32, ..
            } => self.emit_elementwise(context, &lhs, &rhs, &ty, "arith.maximumf")?,
            Typ::Fractional => OperationBuilder::new("arith.maximumf", location)
                .add_operands(&[lhs.1, rhs.1])
                .add_results(&[Type::float64(context)])
                .build()?,
            _ => bail!("np.maximum currently supports float operands only"),
        };
        let value = self
            .block_stack
            .last()
            .context("No active block")?
            .append_operation(operation)
            .result(0)?
            .into();
        Ok((ty, value))
    }

    fn compile_expr(
        &mut self,
        context: &'c Context,
        expr: &ast::Expr,
    ) -> Result<(Typ, Value<'c, 'a>)> {
        match expr {
            ast::Expr::BinOp(binop) => self.compile_binop(context, binop),
            ast::Expr::Name(name) => self.compile_name(context, name),
            ast::Expr::Constant(ast::ExprConstant {
                value: ast::Constant::Float(value),
                ..
            }) => {
                let operation = arith::constant(
                    context,
                    FloatAttribute::new(context, Type::float64(context), *value).into(),
                    ir::Location::unknown(context),
                );
                let value = self
                    .block_stack
                    .last()
                    .context("No active block")?
                    .append_operation(operation)
                    .result(0)?
                    .into();
                Ok((Typ::Fractional, value))
            }
            ast::Expr::UnaryOp(expr)
                if matches!(expr.op, ast::UnaryOp::UAdd | ast::UnaryOp::USub) =>
            {
                let (ty, value) = self.compile_expr(context, &expr.operand)?;
                ensure!(
                    ty == Typ::Fractional,
                    "Unary float signs require a float scalar"
                );
                if expr.op == ast::UnaryOp::UAdd {
                    return Ok((ty, value));
                }
                let value = self
                    .block_stack
                    .last()
                    .context("No active block")?
                    .append_operation(arith::negf(value, ir::Location::unknown(context)))
                    .result(0)?
                    .into();
                Ok((ty, value))
            }
            ast::Expr::Call(call) => self.compile_maximum(context, call),
            _ => bail!("this expression isn't supported at the moment: {:#?}", expr),
        }
    }

    fn compile_function_def(
        &mut self,
        context: &'c Context,
        stmt_def: &ast::StmtFunctionDef,
    ) -> Result<Operation<'c>> {
        self.return_typ = None;
        let mut margs = vec![];
        for arg in stmt_def.args.posonlyargs.iter().chain(&stmt_def.args.args) {
            let myarg = self.emit_arg(context, arg)?;
            margs.push(myarg);
        }
        let loc = ir::Location::unknown(context);
        let block_args: Vec<(Type<'_>, ir::Location)> = margs.iter().map(|a| (*a, loc)).collect();

        let region = Region::new();
        let block = ir::Block::new(&block_args);

        for (index, arg) in stmt_def
            .args
            .posonlyargs
            .iter()
            .chain(&stmt_def.args.args)
            .enumerate()
        {
            let (_, value) = self.variables.get_mut(arg.def.arg.as_str()).unwrap();
            *value = Some(block.argument(index)?.into());
        }

        let block_ref = region.append_block(block);
        self.block_stack.push(block_ref);

        let result = (|| -> Result<Vec<Type<'c>>> {
            for stmt in &stmt_def.body {
                self.compile_stmt(context, stmt)?;
                if block_ref.terminator().is_some() {
                    break;
                }
            }
            if block_ref.terminator().is_none() {
                block_ref.append_operation(func::r#return(&[], loc));
            }
            Ok(block_ref
                .terminator()
                .context("Missing function terminator")?
                .operands()
                .map(|value| value.r#type())
                .collect())
        })();
        // Clear borrowed handles before the region is moved or dropped, including
        // on errors. The returned operation owns the function body.
        self.block_stack.pop();
        for (_, value) in self.variables.values_mut() {
            *value = None;
        }
        let results = result?;
        Ok(func::func(
            context,
            StringAttribute::new(context, stmt_def.name.as_str()),
            TypeAttribute::new(FunctionType::new(context, &margs, &results).into()),
            region,
            &[],
            loc,
        ))
    }

    fn compile_stmtret(&mut self, context: &'c Context, stmt: &ast::StmtReturn) -> Result<()> {
        let block = *self.block_stack.last().context("No active block")?;
        ensure!(
            block.terminator().is_none(),
            "Block already has a terminator"
        );
        let location = ir::Location::unknown(context);
        let (ty, values) = match &stmt.value {
            Some(expr) => {
                let (ty, value) = self.compile_expr(context, expr)?;
                (Some(ty), vec![value])
            }
            None => (None, vec![]),
        };
        block.append_operation(func::r#return(&values, location));
        self.return_typ = ty;
        Ok(())
    }

    fn compile_stmt_assign(&mut self, context: &'c Context, stmt: &ast::StmtAssign) -> Result<()> {
        ensure!(stmt.targets.len() == 1, "expected one assignment");
        let ast::Expr::Name(target) = &stmt.targets[0] else {
            bail!("Only assignments to names are supported");
        };

        let (ty, value) = self.compile_expr(context, &stmt.value)?;
        self.set_variable(target.id.to_string(), ty, value)
    }

    fn compile_stmt(&mut self, context: &'c Context, stmt: &ast::Stmt) -> Result<()> {
        match stmt {
            ast::Stmt::Return(stmt_def) => self.compile_stmtret(context, stmt_def),
            ast::Stmt::Assign(stmt_assign) => self.compile_stmt_assign(context, stmt_assign),
            _ => bail!("not handled, {:#?}", stmt),
        }
    }

    /// Lower the selected function using this call's parameter types.
    /// Imports and other functions are outside this function's lowering scope.
    pub fn compile_module(
        &mut self,
        context: &'c Context,
        module: &ast::Suite,
    ) -> Result<Module<'c>> {
        let function = module
            .iter()
            .find_map(|stmt| match stmt {
                ast::Stmt::FunctionDef(function)
                    if function.name.as_str() == self.function_name =>
                {
                    Some(function)
                }
                _ => None,
            })
            .with_context(|| format!("Function '{}' not found", self.function_name))?;
        let module = Module::new(ir::Location::unknown(context));
        module
            .body()
            .append_operation(self.compile_function_def(context, function)?);
        ensure!(
            module.as_operation().verify(),
            "Generated MLIR failed verification"
        );
        Ok(module)
    }
}

pub fn compile_module<'c>(_context: &'c Context, _module: &ast::Suite) -> Result<Module<'c>> {
    bail!("Compilation requires runtime argument types: construct CompileUnit::from_runtime_args and call unit.compile_module(context, module)")
}

#[cfg(test)]
mod binary_type_tests {
    use super::*;

    fn tensor(shape: &[usize]) -> Typ {
        Typ::Tensor {
            dtype: DType::F32,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn matmul_infers_output_shape_including_empty_dimensions() {
        for (left, right, output) in [
            ([3, 2], [2, 2], [3, 2]),
            ([2, 3], [3, 4], [2, 4]),
            ([0, 3], [3, 4], [0, 4]),
            ([2, 0], [0, 4], [2, 4]),
        ] {
            assert_eq!(
                tensor(&left)
                    .binary_result(ast::Operator::MatMult, &tensor(&right))
                    .unwrap(),
                tensor(&output)
            );
        }
    }

    #[test]
    fn matmul_rejects_bad_dimensions_ranks_dtypes_and_scalars() {
        for (left, right, message) in [
            (
                tensor(&[3, 2]),
                tensor(&[3, 2]),
                "inner dimensions must match",
            ),
            (tensor(&[2]), tensor(&[2, 3]), "rank-2"),
            (tensor(&[1, 3, 2]), tensor(&[2, 4]), "rank-2"),
            (tensor(&[2, 3]), tensor(&[]), "rank-2"),
            (
                tensor(&[2, 3]),
                Typ::Tensor {
                    dtype: DType::F64,
                    shape: vec![3, 4],
                },
                "matching dtypes",
            ),
            (Typ::Integral, Typ::Integral, "requires two matrices"),
            (tensor(&[2, 3]), Typ::Fractional, "requires two matrices"),
        ] {
            let error = left
                .binary_result(ast::Operator::MatMult, &right)
                .unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    #[test]
    fn elementwise_ops_broadcast_shapes_and_require_matching_dtypes() {
        for op in [ast::Operator::Add, ast::Operator::Mult] {
            let matrix = tensor(&[3, 2]);
            assert_eq!(matrix.binary_result(op, &matrix).unwrap(), matrix);
            assert!(matrix
                .binary_result(op, &tensor(&[2, 2]))
                .unwrap_err()
                .to_string()
                .contains("Cannot broadcast"));
            assert_eq!(matrix.binary_result(op, &tensor(&[2])).unwrap(), matrix);
            assert_eq!(
                tensor(&[3, 1]).binary_result(op, &tensor(&[1, 2])).unwrap(),
                matrix
            );
            assert_eq!(
                tensor(&[0, 2]).binary_result(op, &tensor(&[1, 2])).unwrap(),
                tensor(&[0, 2])
            );
            let other_dtype = Typ::Tensor {
                dtype: DType::F64,
                shape: vec![3, 2],
            };
            assert!(matrix
                .binary_result(op, &other_dtype)
                .unwrap_err()
                .to_string()
                .contains("matching dtypes"));
            assert_eq!(
                Typ::Integral.binary_result(op, &Typ::Integral).unwrap(),
                Typ::Integral
            );
            assert_eq!(
                Typ::Fractional.binary_result(op, &Typ::Fractional).unwrap(),
                Typ::Fractional
            );
            assert!(Typ::Integral.binary_result(op, &Typ::Fractional).is_err());
            assert_eq!(matrix.binary_result(op, &Typ::Fractional).unwrap(), matrix);
            assert_eq!(Typ::Fractional.binary_result(op, &matrix).unwrap(), matrix);
        }
    }
}

#[cfg(test)]
mod return_tests {
    use super::*;
    use melior::ir::{
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::FunctionType,
    };

    #[test]
    fn return_statements_produce_valid_function_bodies() {
        for (source, has_value) in [("return x + x", true), ("return", false)] {
            let context = crate::ir::create_context();
            let location = ir::Location::unknown(&context);
            let int_type: Type = IntegerType::new(&context, 64).into();
            let region = Region::new();
            let block = region.append_block(ir::Block::new(&[(int_type, location)]));
            let argument = block.argument(0).unwrap().into();
            let mut unit = CompileUnit {
                function_name: "test".into(),
                variables: HashMap::from([("x".into(), (Typ::Integral, Some(argument)))]),
                block_stack: vec![block],
                return_typ: None,
            };
            let suite = parse_source(source, "test.py").unwrap();
            let ast::Stmt::Return(stmt) = &suite[0] else {
                panic!("expected return statement");
            };
            unit.compile_stmtret(&context, stmt).unwrap();
            assert_eq!(unit.return_typ, has_value.then_some(Typ::Integral));
            assert!(unit
                .compile_stmtret(&context, stmt)
                .unwrap_err()
                .to_string()
                .contains("already has a terminator"));
            drop(unit);

            let results = if has_value { vec![int_type] } else { vec![] };
            let module = Module::new(location);
            module.body().append_operation(func::func(
                &context,
                StringAttribute::new(&context, "test"),
                TypeAttribute::new(FunctionType::new(&context, &[int_type], &results).into()),
                region,
                &[],
                location,
            ));
            assert!(module.as_operation().verify());
        }
    }

    #[test]
    fn return_errors_do_not_emit_a_terminator() {
        let context = crate::ir::create_context();
        let region = Region::new();
        let block = region.append_block(ir::Block::new(&[]));
        let mut unit = CompileUnit {
            function_name: "test".into(),
            variables: HashMap::new(),
            block_stack: vec![],
            return_typ: None,
        };
        let suite = parse_source("return missing", "test.py").unwrap();
        let ast::Stmt::Return(stmt) = &suite[0] else {
            panic!("expected return statement");
        };
        assert!(unit
            .compile_stmtret(&context, stmt)
            .unwrap_err()
            .to_string()
            .contains("No active block"));
        unit.block_stack.push(block);
        assert!(unit
            .compile_stmtret(&context, stmt)
            .unwrap_err()
            .to_string()
            .contains("undefined variable"));
        assert!(block.terminator().is_none());
        assert_eq!(unit.return_typ, None);
    }
}

#[cfg(test)]
mod variable_helpers_tests {
    use super::*;
    use melior::{dialect::arith, ir::BlockLike};

    #[test]
    fn binds_arguments_results_and_reassignments_without_changing_old_ssa_values() {
        let context = crate::ir::create_context();
        let location = ir::Location::unknown(&context);
        let region = Region::new();
        let block = region.append_block(ir::Block::new(&[
            (IntegerType::new(&context, 64).into(), location),
            (IntegerType::new(&context, 1).into(), location),
        ]));
        let argument: Value = block.argument(0).unwrap().into();
        let flag: Value = block.argument(1).unwrap().into();
        let mut unit = CompileUnit {
            function_name: "test".into(),
            variables: HashMap::new(),
            block_stack: vec![block],
            return_typ: None,
        };
        unit.set_variable("x", Typ::Integral, argument).unwrap();
        let x = unit.get_variable("x").unwrap();
        let add = block.append_operation(arith::addi(x, x, location));
        let sum = add.result(0).unwrap().into();
        unit.set_variable("x", Typ::Integral, sum).unwrap();
        assert_eq!(unit.get_variable("x").unwrap(), sum);
        assert_eq!(x, argument);
        assert_ne!(sum, argument);

        unit.set_variable("x", Typ::Boolean, flag).unwrap();
        assert_eq!(unit.variable_type("x"), Some(&Typ::Boolean));
        assert_eq!(unit.get_variable("x").unwrap(), flag);
        assert_eq!(unit.variable_value("x"), Some(flag));
    }

    #[test]
    fn reports_missing_blocks_names_and_unlowered_values() {
        let context = crate::ir::create_context();
        let region = Region::new();
        let block = region.append_block(ir::Block::new(&[(
            IntegerType::new(&context, 64).into(),
            ir::Location::unknown(&context),
        )]));
        let value = block.argument(0).unwrap().into();
        let mut unit = CompileUnit {
            function_name: "test".into(),
            variables: HashMap::from([("x".into(), (Typ::Integral, None))]),
            block_stack: vec![],
            return_typ: None,
        };
        assert!(unit
            .get_variable("x")
            .unwrap_err()
            .to_string()
            .contains("No active block"));
        assert!(unit.set_variable("x", Typ::Integral, value).is_err());
        assert_eq!(unit.variable_value("x"), None);
        unit.block_stack.push(block);
        assert!(unit
            .get_variable("missing")
            .unwrap_err()
            .to_string()
            .contains("Undefined variable"));
        assert!(unit
            .get_variable("x")
            .unwrap_err()
            .to_string()
            .contains("no MLIR value yet"));
        unit.set_variable("x", Typ::Integral, value).unwrap();
        assert_eq!(unit.get_variable("x").unwrap(), value);
    }
}
