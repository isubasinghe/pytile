//! Your exercise: lower a supported Python subset to native MLIR via Melior.
use anyhow::{bail, ensure, Context as _, Result};
use melior::{
    dialect::{arith, func},
    ir::{
        self,
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::{FunctionType, IntegerType},
        BlockLike, BlockRef, Module, Operation, Region, RegionLike, Type, Value, ValueLike,
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
        if lhs.0 != rhs.0 {
            bail!("binary operands must have the same type");
        }
        let location = ir::Location::unknown(context);
        let operation = match (&lhs.0, expr.op) {
            (Typ::Unknown, ast::Operator::Add) => bail!("type was unknown"),
            (Typ::Boolean, ast::Operator::Add) => bail!("adding booleans is not supported"),
            (Typ::Integral, ast::Operator::Add) => arith::addi(lhs.1, rhs.1, location),
            (Typ::Fractional, ast::Operator::Add) => arith::addf(lhs.1, rhs.1, location),
            (ty, op) => bail!("binary operator {op:?} is not supported for {ty:?}"),
        };
        let block = self.block_stack.last().context("No active block")?;
        let result = block.append_operation(operation).result(0)?.into();
        Ok((lhs.0, result))
    }

    fn compile_expr(
        &mut self,
        context: &'c Context,
        expr: &ast::Expr,
    ) -> Result<(Typ, Value<'c, 'a>)> {
        match expr {
            ast::Expr::BinOp(binop) => self.compile_binop(context, binop),
            ast::Expr::Name(name) => self.compile_name(context, name),
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

    fn compile_stmt(&mut self, context: &'c Context, stmt: &ast::Stmt) -> Result<()> {
        match stmt {
            ast::Stmt::Return(stmt_def) => self.compile_stmtret(context, stmt_def),
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
