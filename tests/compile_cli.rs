use melior::ir::{operation::OperationLike, Module};
use pytile::{
    frontend::{parse_source, Typ},
    runtime::unit_from_expression,
};
use rumpy::DType;
use rustpython_parser::ast;
use std::{
    fs,
    process::{Command, Output},
};

fn compile(
    source: &std::path::Path,
    output: &std::path::Path,
    args: &str,
    function: Option<&str>,
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pytile"));
    command
        .arg("compile")
        .arg(source)
        .arg("--args")
        .arg(args)
        .arg("-o")
        .arg(output);
    if let Some(function) = function {
        command.arg("--function").arg(function);
    }
    command.output().unwrap()
}

#[test]
fn compile_emits_verified_integer_add_without_executing_source() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("add.py");
    let output = dir.path().join("add.mlir");
    fs::write(
        &source,
        "raise RuntimeError('must not run')\ndef compute(x, y):\n    return x + y\n",
    )
    .unwrap();
    let result = compile(&source, &output, "(2, 3)", None);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = fs::read_to_string(&output).unwrap();
    let context = pytile::ir::create_context();
    assert!(Module::parse(&context, &text)
        .unwrap()
        .as_operation()
        .verify());
    assert!(text.contains("func.func @compute(%arg0: i64, %arg1: i64) -> i64"));
    assert!(text.contains("arith.addi %arg0, %arg1 : i64"));
    assert!(text.contains("return %0 : i64"));

    // Unsupported inputs must leave a previously generated artifact intact.
    let result = compile(
        &source,
        &output,
        "(np.zeros(128, dtype=np.float64), np.ones(128, dtype=np.float64))",
        None,
    );
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("is not supported yet"));
    assert_eq!(fs::read_to_string(&output).unwrap(), text);
}

#[test]
fn float32_matrix_arguments_and_returns_preserve_shape() {
    let context = pytile::ir::create_context();
    let ast = parse_source("def identity(x): return x\n", "identity.py").unwrap();
    let ast::Stmt::FunctionDef(function) = &ast[0] else {
        panic!("expected function");
    };
    let mut unit = unit_from_expression(function, "(np.zeros((3, 2), dtype=np.float32),)").unwrap();
    let module = unit.compile_module(&context, &ast).unwrap();
    assert!(module.as_operation().verify());
    let text = module.as_operation().to_string();
    assert!(text.contains("@identity(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32>"));
    assert!(text.contains("return %arg0 : tensor<3x2xf32>"));
}

#[test]
fn matmul_lowers_rectangular_matrices_and_rejects_invalid_shapes() {
    let context = pytile::ir::create_context();
    let ast = parse_source("def compute(x, y): return x @ y\n", "matmul.py").unwrap();
    let ast::Stmt::FunctionDef(function) = &ast[0] else {
        panic!("expected function");
    };
    for (left_shape, right_shape) in [("(3, 2)", "(2, 4)"), ("(3, 0)", "(0, 4)")] {
        let args = format!(
            "(np.zeros({left_shape}, dtype=np.float32), np.zeros({right_shape}, dtype=np.float32))"
        );
        let mut unit = unit_from_expression(function, &args).unwrap();
        let module = unit.compile_module(&context, &ast).unwrap();
        assert!(module.as_operation().verify());
        let text = module.as_operation().to_string();
        assert!(text.contains("linalg.fill"));
        assert!(text.contains("linalg.matmul"));
        assert!(text.contains("-> tensor<3x4xf32>"));
    }
    let mut unit = unit_from_expression(
        function,
        "(np.zeros((3, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float32))",
    )
    .unwrap();
    let error = unit.compile_module(&context, &ast).err().unwrap();
    assert!(
        error.to_string().contains("inner dimensions must match"),
        "{error}"
    );
}

#[test]
fn compile_validates_input_expressions_and_argument_counts() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("add.py");
    let output = dir.path().join("add.mlir");
    fs::write(&source, include_str!("../examples/add.py")).unwrap();
    for (args, message) in [
        ("[2, 3]", "must evaluate to a tuple"),
        ("(2,)", "expected 2 explicit positional arguments"),
        ("(missing, 3)", "NameError"),
        ("(", "Invalid --args expression"),
    ] {
        let result = compile(&source, &output, args, None);
        assert!(!result.status.success());
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(stderr.contains(message), "{stderr}");
        assert!(!output.exists());
    }
}

#[test]
fn compile_requires_selection_when_multiple_functions_exist() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("functions.py");
    let output = dir.path().join("result.mlir");
    fs::write(
        &source,
        "def first(x): return x\ndef second(x, y): return x + y\n",
    )
    .unwrap();
    let result = compile(&source, &output, "(2, 3)", None);
    assert!(String::from_utf8_lossy(&result.stderr).contains("select one with --function"));
    let result = compile(&source, &output, "(2, 3)", Some("second"));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = fs::read_to_string(&output).unwrap();
    assert!(text.contains("func.func @second"));
    assert!(!text.contains("@first"));
    let result = compile(&source, &output, "(2, 3)", Some("missing"));
    assert!(String::from_utf8_lossy(&result.stderr).contains("function named 'missing'"));
}

#[test]
fn lowering_clears_borrowed_values_after_success_and_failure() {
    let context = pytile::ir::create_context();
    let source = "def compute(x, y): return missing\n";
    let broken = parse_source(source, "test.py").unwrap();
    let ast::Stmt::FunctionDef(function) = &broken[0] else {
        panic!("expected function");
    };
    let mut unit = unit_from_expression(function, "(2, 3)").unwrap();
    let before = unit.variables().clone();
    assert!(unit.compile_module(&context, &broken).is_err());
    assert_eq!(unit.variables(), &before);

    for source in [
        include_str!("../examples/add.py"),
        "def compute(x, y):\n    return\n    return missing\n",
    ] {
        let ast = parse_source(source, "test.py").unwrap();
        let module = unit.compile_module(&context, &ast).unwrap();
        assert!(module.as_operation().verify());
        drop(module);
        assert_eq!(unit.variables(), &before);
    }
}

#[test]
fn evaluated_argument_metadata_outlives_the_interpreter() {
    let module = parse_source(include_str!("../examples/add.py"), "add.py").unwrap();
    let ast::Stmt::FunctionDef(function) = &module[0] else {
        panic!("expected function")
    };
    let unit =
        unit_from_expression(function, "(np.zeros((4, 784), dtype=np.float32), 1.5)").unwrap();
    assert_eq!(
        unit.variable_type("x"),
        Some(&Typ::Tensor {
            dtype: DType::F32,
            shape: vec![4, 784]
        })
    );
    assert_eq!(unit.variable_type("y"), Some(&Typ::Fractional));
}
