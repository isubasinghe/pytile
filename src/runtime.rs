//! Construct a compilation signature from sample inputs evaluated by RustPython.
use anyhow::{anyhow, ensure, Result};
use rustpython_parser::ast;
use rustpython_vm::{builtins::PyTuple, compiler::Mode, Interpreter};

use crate::frontend::CompileUnit;

/// Evaluate an explicit tuple of sample arguments with Rumpy available as `np`.
/// The target function is not executed. The returned unit owns only metadata.
pub fn unit_from_expression<'c, 'a>(
    function: &ast::StmtFunctionDef,
    expression: &str,
) -> Result<CompileUnit<'c, 'a>> {
    let builder = Interpreter::builder(Default::default());
    let numpy = rumpy::module_def(&builder.ctx);
    let interpreter = builder.add_native_module(numpy).build();
    let result = interpreter.enter(|vm| -> Result<CompileUnit<'c, 'a>> {
        let scope = vm.new_scope_with_builtins();
        let setup = vm
            .compile("import numpy as np", Mode::Exec, "<input setup>".into())
            .map_err(|error| anyhow!("{error}"))?;
        let format_error = |error: rustpython_vm::builtins::PyBaseExceptionRef| {
            let mut message = String::new();
            let _ = vm.write_exception(&mut message, &error);
            anyhow!("Evaluating sample arguments:\n{message}")
        };
        vm.run_code_obj(setup, scope.clone())
            .map_err(format_error)?;
        let code = vm
            .compile(expression, Mode::Eval, "<--args>".into())
            .map_err(|error| anyhow!("Invalid --args expression: {error}"))?;
        let arguments = vm.run_code_obj(code, scope).map_err(format_error)?;
        let arguments = arguments.downcast_ref::<PyTuple>().ok_or_else(|| {
            anyhow!("--args must evaluate to a tuple, for example '(2, 3)' or '(2,)'")
        })?;
        CompileUnit::from_runtime_args(function, arguments.as_slice(), vm)
    });
    let exit_code = interpreter.finalize(None);
    let unit = result?;
    ensure!(exit_code == 0, "Interpreter shutdown failed: {exit_code}");
    Ok(unit)
}
