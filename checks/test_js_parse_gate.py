"""Discriminating test: malformed JS nested under static/fork MUST fail CI.

The fixture creates real .js files and invokes the exact CI script. The negative
control is intentionally not a source-string assertion.
"""
from pathlib import Path
import subprocess
import sys

GATE = Path(__file__).resolve().parent / "js_parse_gate.py"


def _run(root):
    return subprocess.run([sys.executable, str(GATE), "--root", str(root)],
                          text=True, capture_output=True, check=False, timeout=25)


def test_recursive_gate_accepts_valid_browser_code(tmp_path):
    parent = tmp_path / "static"
    nested = parent / "fork"
    nested.mkdir(parents=True)
    (parent / "app.js").write_text("const foo = 2;\n")
    (nested / "model-library.js").write_text("window.sample = () => 3;\n")
    result = _run(parent)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 files checked, 0 failures" in result.stdout


def test_recursive_gate_rejects_real_syntax_error_in_nested_fork(tmp_path):
    parent = tmp_path / "static"
    nested = parent / "fork"
    nested.mkdir(parents=True)
    (parent / "app.js").write_text("const foo = 2;\n")
    (nested / "model-library.js").write_text("function invalid( {\n")
    result = _run(parent)
    assert result.returncode != 0, "A broken fork extension escaped the canonical JS gate"
    assert "model-library.js" in result.stderr
    assert "2 files checked, 1 failures" in result.stdout


def test_recursive_gate_rejects_vacuous_green(tmp_path):
    assert _run(tmp_path).returncode != 0
    assert _run(tmp_path / "absent").returncode != 0
