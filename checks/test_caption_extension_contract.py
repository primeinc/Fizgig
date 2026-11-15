"""Behavioral and load-order contracts for the fork-owned caption rule UI.

The executable CJS test loads the real extension with a mocked server that
distinguishes successful persistence from a failed write and checks that
saved rules are scoped to the current dataset folder.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_caption_rules_extension_under_real_node_vm():
    result = subprocess.run(
        ["node", "checks/caption_rules_contract.cjs"],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "folder isolation, failing-save gate" in result.stdout


def test_caption_fork_assets_are_loaded_before_upstream_app():
    html = (ROOT / "src/fizgig/web/static/index.html").read_text(encoding="utf-8")
    upstream = (ROOT / "src/fizgig/web/static/app.js").read_text(encoding="utf-8")
    extension = (ROOT / "src/fizgig/web/static/fork/caption-rules.js").read_text(encoding="utf-8")
    assert html.index('src="/fork/caption-rules.js"') < html.index('src="/app.js"')
    assert html.index('href="/style.css"') < html.index('href="/fork/caption-rules.css"')
    assert 'window.FizgigForkCaptionRules' in upstream
    assert 'window.FizgigForkCaptionRules' in extension
    assert "function renderCaptionRules()" not in upstream
