use std::{fs, path::Path};

use pytile::backend::compile_ptx;

#[test]
#[ignore = "requires MLIR/LLVM tools; run make test-backend in the full nix shell"]
fn reference_kernel_compiles_to_sm75() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("smoke.ptx");
    compile_ptx(Path::new("examples/smoke.mlir"), &output).unwrap();
    let ptx = fs::read_to_string(output).unwrap();
    assert!(ptx.contains(".target sm_75"));
    assert!(ptx.contains(".entry compute("));
    assert!(ptx.contains("add.rn.f32"));
    assert!(ptx.contains("mul.rn.f32"));
}

#[test]
#[ignore = "requires MLIR/LLVM tools; run make test-backend in the full nix shell"]
fn invalid_ir_fails_without_ptx() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("invalid.mlir");
    fs::write(&source, "gpu.module @broken { invalid.operation }").unwrap();
    let output = dir.path().join("output.ptx");
    assert!(compile_ptx(&source, &output).is_err());
    assert!(!output.exists());
}

#[test]
fn input_is_not_overwritten_by_output_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("kernel.nvvm.mlir");
    fs::write(&source, "preserve me").unwrap();
    let error = compile_ptx(&source, &dir.path().join("kernel.ptx")).unwrap_err();
    assert!(error.to_string().contains("overlaps"));
    assert_eq!(fs::read_to_string(source).unwrap(), "preserve me");
}
