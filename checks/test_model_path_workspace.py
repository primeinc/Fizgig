"""Exercise the fork model library at its real upstream Preferences render seam.

The implementation is not protected by source-string equality tests: Node executes
the current upstream renderer and the separate fork adapter against a real model
family, a synthetic future family, and deliberate failing controls.
"""
import json
import subprocess
from pathlib import Path

from fizgig.web.schema import get_schema

ROOT = Path(__file__).resolve().parents[1]


def test_schema_driven_fork_library_renders_without_upstream_model_overrides():
    family = next(f for f in get_schema()["families"] if f["key"] == "minimax_image")
    result = subprocess.run(
        ["node", "checks/model_library_contract.cjs"],
        cwd=ROOT, input=json.dumps(family), text=True,
        capture_output=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "fallback and negative controls" in result.stdout


def test_fork_assets_are_explicitly_served_and_included_in_page():
    """The browser must receive the extension before app.js evaluates boot()."""
    html = (ROOT / "src/fizgig/web/static/index.html").read_text(encoding="utf-8")
    server = (ROOT / "src/fizgig/web/server.py").read_text(encoding="utf-8")
    assert html.index("/fork-model-library.js") < html.index('src="/app.js"')
    assert html.index("/style.css") < html.index("/fork-model-library.css")
    assert server.count('"/fork-model-library.js"') == 1
    assert server.count('"/fork-model-library.css"') == 1
