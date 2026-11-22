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
fn compile_reaches_typed_lowering_without_executing_source_or_clobbering_output() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("add.py");
    let output = dir.path().join("add.mlir");
    fs::write(
        &source,
        "raise RuntimeError('must not run')\ndef compute(x, y):\n    return x + y\n",
    )
    .unwrap();
    fs::write(&output, "keep existing output").unwrap();
    for args in [
        "(2, 3)",
        "(np.zeros(128, dtype=np.float32), np.ones(128, dtype=np.float32))",
    ] {
        let result = compile(&source, &output, args, None);
        assert!(!result.status.success()); // Lowering intentionally remains the exercise.
        assert!(String::from_utf8_lossy(&result.stderr).contains("lowering is not implemented"));
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep existing output");
    }
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
    assert!(String::from_utf8_lossy(&result.stderr).contains("lowering is not implemented"));
    let result = compile(&source, &output, "(2, 3)", Some("missing"));
    assert!(String::from_utf8_lossy(&result.stderr).contains("function named 'missing'"));
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
