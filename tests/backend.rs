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
fn scalar_add_compiles_to_cuda_kernel() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("input.mlir");
    let output = dir.path().join("add.ptx");
    fs::write(
        &source,
        "module { func.func @compute(%x: i64, %y: i64) -> i64 {
        %sum = arith.addi %x, %y : i64
        return %sum : i64
    } }",
    )
    .unwrap();
    compile_ptx(&source, &output).unwrap();
    let ptx = fs::read_to_string(output).unwrap();
    assert!(ptx.contains(".target sm_75"));
    assert!(ptx.contains(".entry compute_kernel("));
    assert!(ptx.contains("add.s64"));
    // Bare pointers may have only byte alignment, so LLVM can split an i64
    // result into two 32-bit stores.
    assert!(ptx.contains("st.b32") || ptx.contains("st.u64") || ptx.contains("st.global.u64"));
}

#[test]
#[ignore = "requires MLIR/LLVM tools; run make test-backend in the full nix shell"]
fn mnist_forward_compiles_to_ordered_gpu_kernels() {
    use pytile::{frontend::parse_source, runtime::unit_from_expression};
    use rustpython_parser::ast;

    let context = pytile::ir::create_context();
    let ast = parse_source(include_str!("../examples/mnist_forward.py"), "mnist.py").unwrap();
    let function = ast
        .iter()
        .find_map(|stmt| match stmt {
            ast::Stmt::FunctionDef(f) => Some(f),
            _ => None,
        })
        .unwrap();
    let mut unit = unit_from_expression(function,
        "tuple(np.zeros(shape, dtype=np.float32) for shape in [(3, 784), (784, 32), (32,), (32, 10), (10,)])").unwrap();
    let module = unit.compile_module(&context, &ast).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("mnist.mlir");
    let output = dir.path().join("mnist.ptx");
    fs::write(&source, module.as_operation().to_string()).unwrap();
    compile_ptx(&source, &output).unwrap();
    let ptx = fs::read_to_string(&output).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(output.with_extension("json")).unwrap()).unwrap();
    assert_eq!(manifest["inputs"], serde_json::json!([0, 1, 2, 3, 4]));
    let output_id = manifest["output"].as_u64().unwrap() as usize;
    assert_eq!(manifest["buffers"][output_id], serde_json::json!([3, 10]));
    assert_eq!(manifest["kernels"].as_array().unwrap().len(), 7);
    assert_eq!(ptx.matches(".entry ").count(), 7);
    assert_eq!(manifest["ptx_fnv1a64"].as_str().unwrap().len(), 16);
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
