//! Your exercise: lower a supported Python subset to native MLIR via Melior.
use anyhow::{bail, ensure, Context as _, Result};
use melior::{Context, ir::{self, BlockLike, BlockRef, Module, Region, RegionLike, Type, Value, r#type::IntegerType}};
use rumpy::{DType, PyNdArray};
use rustpython_parser::{
    ast::{self},
    Parse,
};
use rustpython_vm::{
    AsObject, PyObjectRef, VirtualMachine, builtins::{PyFloat, PyInt}, compiler::ast::NodeKind::ExprName
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
    return_typ: Option<Typ>
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
            block_stack: vec![]
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
        let (_, value) = self.variables.get(name)
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

    fn emit_arg(
        &mut self,
        context: &'c Context,
        arg: &ast::ArgWithDefault,
    ) -> Result<Type<'c>> {
        let (ty, _value) = self
            .variables
            .get(arg.def.arg.as_str())
            .with_context(|| format!("Missing runtime type for '{}'", arg.def.arg))?;
        
        let mty = match ty {
            Typ::Integral => IntegerType::new(context, 64).into(),
            Typ::Unknown => panic!("unknown types are not supported at the moment"),
            _ => bail!("this type: {:#?}  is not supported", ty)
        };
        Ok(mty)
    }

    fn compile_name(&mut self, context: &'c Context, name: &ast::ExprName) -> Result<(Typ, Value<'c, 'a>)> {
        todo!();
    }
    fn compile_binop(&mut self, context: &'c Context, expr: &ast::ExprBinOp) -> Result<(Typ, Value<'c, 'a>)> {
        println!("expr: {:#?}", expr);
        let lhs = self.compile_expr(context, &expr.left)?;
        let rhs = self.compile_expr(context, &expr.right)?;
        match expr.op {
            ast::Operator::Add => panic!("not implemented"),
            _ => panic!("not implemented")
        };
        todo!()
    }

    fn compile_expr(&mut self, context: &'c Context, expr: &ast::Expr) -> Result<(Typ, Value<'c, 'a>)>{ 
        match expr {
            ast::Expr::BinOp(binop) => self.compile_binop(context, binop),
            ast::Expr::Name(name) => self.compile_name(context, name),
            _ => bail!("this expression isn't supported at the moment: {:#?}", expr)
        }
    }

    fn compile_function_def(
        &mut self,
        context: &'c Context,
        stmt_def: &ast::StmtFunctionDef,
    ) -> Result<()> {
        let mut margs = vec![];
        for arg in stmt_def.args.posonlyargs.iter().chain(&stmt_def.args.args) {
            let myarg = self.emit_arg(context, arg)?;
            margs.push(myarg);
        }
        let loc = ir::Location::unknown(context);
        let block_args: Vec<(Type<'_>, ir::Location)>= margs.iter().map(|a| { (*a, loc) }).collect();

        let region = Region::new();
        let block = ir::Block::new(&block_args);

        for (index, arg) in stmt_def.args.posonlyargs.iter().chain(&stmt_def.args.args).enumerate() {
            let a = &arg.as_arg().arg;
            let (typ, _)  = self.variables.get(&a.to_string()).unwrap().clone();
            let s = &arg.as_arg().arg.to_string();
            let block_arg = block.argument(index)?;
            let val = Some(block_arg.into());
            self.variables.insert(s.to_string(), (typ, val));
        }

        let block_ref = region.append_block(block);
        self.block_stack.push(block_ref);

        for stmt in &stmt_def.body {
            self.compile_stmt(context, stmt)?;
        }
        if let Some(retexpr) = &stmt_def.returns {
        };
        Ok(())
    }

    fn compile_stmtret(&mut self, context: &'c Context, stmt: &ast::StmtReturn) -> Result<()> {
        if stmt.value.is_none() {
            return Ok(());
        }
        let expr = stmt.value.as_ref().unwrap();
        self.compile_expr(context, expr)?;
        Ok(())
    }

    fn compile_stmt(&mut self, context: &'c Context, stmt: &ast::Stmt) -> Result<()> {
        match stmt {
            ast::Stmt::FunctionDef(stmt_def) => self.compile_function_def(context, stmt_def),
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
        let statement = module
            .iter()
            .find(|stmt| {
                matches!(stmt,
            ast::Stmt::FunctionDef(function) if function.name.as_str() == self.function_name)
            })
            .with_context(|| format!("Function '{}' not found", self.function_name))?;
        self.compile_stmt(context, statement)?;
        bail!("Python-to-MLIR lowering is not implemented yet; runtime parameter types are available in CompileUnit")
    }
}

pub fn compile_module<'c>(_context: &'c Context, _module: &ast::Suite) -> Result<Module<'c>> {
    bail!("Compilation requires runtime argument types: construct CompileUnit::from_runtime_args and call unit.compile_module(context, module)")
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
            function_name: "test".into(), variables: HashMap::new(),
            block_stack: vec![block], return_typ: None,
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
        let block = region.append_block(ir::Block::new(&[
            (IntegerType::new(&context, 64).into(), ir::Location::unknown(&context)),
        ]));
        let value = block.argument(0).unwrap().into();
        let mut unit = CompileUnit {
            function_name: "test".into(),
            variables: HashMap::from([("x".into(), (Typ::Integral, None))]),
            block_stack: vec![], return_typ: None,
        };
        assert!(unit.get_variable("x").unwrap_err().to_string().contains("No active block"));
        assert!(unit.set_variable("x", Typ::Integral, value).is_err());
        assert_eq!(unit.variable_value("x"), None);
        unit.block_stack.push(block);
        assert!(unit.get_variable("missing").unwrap_err().to_string().contains("Undefined variable"));
        assert!(unit.get_variable("x").unwrap_err().to_string().contains("no MLIR value yet"));
        unit.set_variable("x", Typ::Integral, value).unwrap();
        assert_eq!(unit.get_variable("x").unwrap(), value);
    }
}
