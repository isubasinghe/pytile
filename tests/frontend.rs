use std::{fs, process::Command};

use pytile::frontend::parse_source;

#[test]
fn parsing_does_not_execute_python() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("must-not-exist");
    let source = format!(
        "open({:?}, 'w').write('unexpected')",
        marker.to_str().unwrap()
    );
    assert!(parse_source(&source, "input.py").is_ok());
    assert!(!marker.exists());
}

#[test]
fn cli_reports_python_syntax_errors() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("broken.py");
    fs::write(&source, "def broken(:\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_pytile"))
        .arg("ast")
        .arg(source)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("broken.py"));
}

#[test]
fn cli_dumps_python_ast() {
    let result = Command::new(env!("CARGO_BIN_EXE_pytile"))
        .args(["ast", "examples/square_sum.py"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(text.contains("FunctionDef"));
    assert!(text.contains("BinOp"));
}
