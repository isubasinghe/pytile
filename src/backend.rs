//! Adapt scalar MLIR to GPU kernels and invoke the standard MLIR/LLVM tools.

mod scalar;
mod tensor;

use std::{fs, path::Path, process::Command};

use anyhow::{bail, ensure, Context, Result};

fn run(command: &mut Command) -> Result<()> {
    let tool = command.get_program().to_string_lossy().into_owned();
    let status = command
        .status()
        .with_context(|| format!("Could not run {tool}; enter `nix develop` first"))?;
    ensure!(status.success(), "{tool} failed with {status}");
    Ok(())
}

pub fn compile_ptx(source: &Path, output: &Path) -> Result<()> {
    let source = source
        .canonicalize()
        .context("Cannot locate the source MLIR")?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let output = parent
        .canonicalize()?
        .join(output.file_name().context("Missing output filename")?);
    let verified = output.with_extension("verified.mlir");
    let gpu = output.with_extension("gpu.mlir");
    let lowered = output.with_extension("nvvm.mlir");
    let device = output.with_extension("device.mlir");
    let llvm_ir = output.with_extension("ll");
    let manifest_path = output.with_extension("json");
    let artifacts = [
        &output,
        &verified,
        &gpu,
        &lowered,
        &device,
        &llvm_ir,
        &manifest_path,
    ];
    for (index, path) in artifacts.iter().enumerate() {
        let resolved = path.canonicalize().unwrap_or_else(|_| (*path).clone());
        ensure!(resolved != source, "Input MLIR overlaps an output artifact");
        for previous in &artifacts[..index] {
            let previous = previous
                .canonicalize()
                .unwrap_or_else(|_| (*previous).clone());
            ensure!(
                resolved != previous,
                "Output artifacts overlap; use a .ptx output filename"
            );
        }
    }

    // Parse exactly one root operation, then restrict the public backend input.
    run(Command::new("mlir-opt")
        .arg(&source)
        .args(["--no-implicit-module", "-o"])
        .arg(&verified))?;
    let text = fs::read_to_string(&verified)?;
    let mut manifest = None;
    let device_input = if text.trim_start().starts_with("gpu.module ") {
        &verified
    } else {
        let gpu_source = if let Some(program) = tensor::lower(&text)? {
            manifest = Some(program.manifest);
            program.gpu
        } else {
            scalar::wrap(&text)?
        };
        fs::write(&gpu, gpu_source)?;
        &gpu
    };

    // LLVM 21 requires a parent builtin.module during GPU data-layout lookup.
    let pipeline = "builtin.module(arith-expand,gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{use-bare-ptr-memref-call-conv=true}),convert-func-to-llvm,reconcile-unrealized-casts)";
    run(Command::new("mlir-opt")
        .arg(device_input)
        .arg(format!("--pass-pipeline={pipeline}"))
        .arg("-o")
        .arg(&lowered))?;
    let text = fs::read_to_string(&lowered)?;
    let lines: Vec<_> = text.trim().lines().collect();
    if lines.len() < 4
        || lines[0] != "module {"
        || !lines[1].starts_with("  gpu.module ")
        || lines[lines.len() - 2..] != ["  }", "}"]
    {
        bail!("Unexpected MLIR printer output; expected one wrapped gpu.module");
    }
    // Export the device module, not the parent (which would yield empty LLVM IR).
    fs::write(
        &device,
        format!("{}\n", lines[1..lines.len() - 1].join("\n")),
    )?;
    run(Command::new("mlir-translate")
        .arg(&device)
        .args(["--no-implicit-module", "--mlir-to-llvmir", "-o"])
        .arg(&llvm_ir))?;
    run(Command::new("llc")
        .args([
            "-mtriple=nvptx64-nvidia-cuda",
            "-mcpu=sm_75",
            "-mattr=+ptx75",
            "-O3",
        ])
        .arg(&llvm_ir)
        .arg("-o")
        .arg(&output))?;
    if let Some(mut manifest) = manifest {
        let fingerprint = fs::read(&output)?
            .iter()
            .fold(0xcbf29ce484222325_u64, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            });
        manifest.ptx_fnv1a64 = format!("{fingerprint:016x}");
        fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    }
    Ok(())
}
