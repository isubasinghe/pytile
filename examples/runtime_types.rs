//! Inspect real RustPython/Rumpy inputs without running the target function.
use anyhow::{anyhow, Result};
use pytile::frontend::{parse_source, CompileUnit};
use rustpython_parser::ast;
use rustpython_vm::{builtins::PyTuple, compiler::Mode, Interpreter, VirtualMachine};

fn inspect(vm: &VirtualMachine, source: &str, name: &str, inputs: &str) -> Result<()> {
    let module = parse_source(source, name)?;
    let function = module
        .iter()
        .find_map(|stmt| match stmt {
            ast::Stmt::FunctionDef(function) if function.name.as_str() == name => Some(function),
            _ => None,
        })
        .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
    let scope = vm.new_scope_with_builtins();
    let code = vm
        .compile(inputs, Mode::Exec, "<runtime inputs>".into())
        .map_err(|error| anyhow!("{error}"))?;
    vm.run_code_obj(code, scope.clone()).map_err(|error| {
        let mut message = String::new();
        let _ = vm.write_exception(&mut message, &error);
        anyhow!(message)
    })?;
    let args = scope
        .globals
        .get_item("args", vm)
        .map_err(|_| anyhow!("Missing args"))?;
    let args = args
        .downcast_ref::<PyTuple>()
        .ok_or_else(|| anyhow!("Expected args tuple"))?;
    let unit = CompileUnit::from_runtime_args(function, args.as_slice(), vm)?;
    println!("{name}:");
    for arg in function.args.posonlyargs.iter().chain(&function.args.args) {
        println!(
            "  {}: {:?}",
            arg.def.arg,
            unit.variable_type(arg.def.arg.as_str()).unwrap()
        );
    }
    // Next step, once you implement lowering:
    // let mut unit = unit;
    // let mlir = unit.compile_module(&context, &module)?;
    Ok(())
}

fn main() -> Result<()> {
    let builder = Interpreter::builder(Default::default());
    let numpy = rumpy::module_def(&builder.ctx);
    let interpreter = builder.add_native_module(numpy).build();
    let result = interpreter.enter(|vm| -> Result<()> {
        inspect(vm, include_str!("add.py"), "compute", "args = (2, 3)")?;
        inspect(vm, include_str!("add.py"), "compute", "import numpy as np\nargs = (np.zeros((128,), dtype=np.float32), np.ones((128,), dtype=np.float32))")?;
        inspect(vm, include_str!("mnist_forward.py"), "forward", r#"
import numpy as np
args = (
    np.zeros((4, 784), dtype=np.float32),
    np.zeros((784, 32), dtype=np.float32),
    np.zeros((32,), dtype=np.float32),
    np.zeros((32, 10), dtype=np.float32),
    np.zeros((10,), dtype=np.float32),
)
"#)
    });
    let exit_code = interpreter.finalize(None);
    result?;
    anyhow::ensure!(exit_code == 0, "Interpreter shutdown failed: {exit_code}");
    Ok(())
}
