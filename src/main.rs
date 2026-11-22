use std::{fs, path::PathBuf};

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use melior::ir::operation::OperationLike;
use rustpython_parser::ast;

#[derive(Parser)]
#[command(version, about = "Compile a Python subset through MLIR to NVIDIA PTX")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse Python in Rust and display its syntax tree.
    Ast { source: PathBuf },
    /// Emit high-level func/arith MLIR (Python lowering is your exercise).
    Compile {
        source: PathBuf,
        /// Target function; inferred when the file defines exactly one function.
        #[arg(long)]
        function: Option<String>,
        /// Python tuple of sample inputs; Rumpy is available as np.
        #[arg(long, value_name = "PYTHON_TUPLE")]
        args: String,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Lower a handwritten or generated GPU MLIR module to sm_75 PTX.
    Ptx {
        source: PathBuf,
        #[arg(short, long, default_value = "build/kernel.ptx")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Ast { source } => {
            let text = fs::read_to_string(&source)
                .with_context(|| format!("Cannot read {}", source.display()))?;
            let module = pytile::frontend::parse_source(&text, &source.to_string_lossy())?;
            println!("{module:#?}");
        }
        Commands::Compile {
            source,
            function,
            args,
            output,
        } => {
            let text = fs::read_to_string(&source)
                .with_context(|| format!("Cannot read {}", source.display()))?;
            let module = pytile::frontend::parse_source(&text, &source.to_string_lossy())?;
            let functions: Vec<_> = module
                .iter()
                .filter_map(|statement| match statement {
                    ast::Stmt::FunctionDef(definition) => Some(definition),
                    _ => None,
                })
                .collect();
            let definition = if let Some(name) = function {
                let matches: Vec<_> = functions
                    .iter()
                    .copied()
                    .filter(|definition| definition.name.as_str() == name)
                    .collect();
                ensure!(
                    matches.len() == 1,
                    "Expected exactly one top-level function named '{name}', found {}",
                    matches.len()
                );
                matches[0]
            } else {
                ensure!(
                    functions.len() == 1,
                    "Expected one top-level function, found {}; select one with --function",
                    functions.len()
                );
                functions[0]
            };
            // Declare the owner first so the unit and its block references drop first.
            let context = pytile::ir::create_context();
            let mut unit = pytile::runtime::unit_from_expression(definition, &args)?;
            let mlir = unit.compile_module(&context, &module)?;
            ensure!(
                mlir.as_operation().verify(),
                "Generated MLIR failed verification"
            );
            if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            fs::write(&output, mlir.as_operation().to_string())?;
            println!("{}", output.display());
        }
        Commands::Ptx { source, output } => {
            pytile::backend::compile_ptx(&source, &output)?;
            println!("{}", output.display());
        }
    }
    Ok(())
}
