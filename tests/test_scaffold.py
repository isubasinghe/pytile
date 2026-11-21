"""Scaffold checks; add compiler acceptance tests as you implement lowering."""

import ast
import subprocess
import sys

from pytile.frontend import parse_source


def test_parsing_does_not_execute_source(tmp_path):
    marker = tmp_path / "must-not-exist"
    module = parse_source(f"open({str(marker)!r}, 'w').write('unexpected')")
    assert isinstance(module, ast.Module)
    assert not marker.exists()


def test_cli_reports_syntax_error(tmp_path):
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytile", str(source), "--dump-ast"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "broken.py" in result.stderr


def test_cli_dumps_ast():
    result = subprocess.run(
        [sys.executable, "-m", "pytile", "examples/add.py", "--dump-ast"],
        capture_output=True, text=True, check=True,
    )
    assert "FunctionDef" in result.stdout
    assert "Add()" in result.stdout
