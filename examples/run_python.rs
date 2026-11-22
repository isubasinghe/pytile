//! Run with `cargo run --example run_python -- examples/add.py`.
//! `--compile-only` parses and compiles to Python bytecode without executing imports.
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rustpython_vm::{compiler::Mode, Interpreter};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Check syntax and compile to RustPython bytecode without execution.
    #[arg(long)]
    compile_only: bool,
    /// Files execute in order, sharing a global scope.
    #[arg(required = true)]
    files: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let builder = Interpreter::builder(Default::default());
    let numpy = rumpy::module_def(&builder.ctx);
    let interpreter = builder.add_native_module(numpy).build();
    let result = interpreter.enter(|vm| -> Result<()> {
        let scope = vm.new_scope_with_builtins();
        for path in &args.files {
            let source = std::fs::read_to_string(path)
                .with_context(|| format!("Reading {}", path.display()))?;
            let code = vm
                .compile(&source, Mode::Exec, path.display().to_string())
                .map_err(|error| anyhow!("{}: {error}", path.display()))?;
            println!("Compiled {} to RustPython bytecode", path.display());
            if !args.compile_only {
                vm.run_code_obj(code, scope.clone()).map_err(|error| {
                    let mut message = String::new();
                    let _ = vm.write_exception(&mut message, &error);
                    anyhow!("Executing {}:\n{message}", path.display())
                })?;
                println!("Executed {}", path.display());
            }
        }
        Ok(())
    });
    let exit_code = interpreter.finalize(None);
    result?;
    anyhow::ensure!(exit_code == 0, "Interpreter shutdown failed: {exit_code}");
    Ok(())
}
