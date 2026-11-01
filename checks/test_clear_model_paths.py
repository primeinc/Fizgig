"""Model-path Clear actions are per-field and do not delete model files.

The browser uses POST /api/prefs with changes {model_key: ""}; test the same
persistence operation and pin the current UI wiring so model rows keep a
visible Clear action alongside Choose and Download.
"""
from pathlib import Path

from fizgig.web import server


def test_clear_only_the_selected_model_path(tmp_path, monkeypatch):
    prefs_file = tmp_path / "user_data" / "prefs.json"
    prefs_file.parent.mkdir()
    dit = tmp_path / "models" / "qwen_image_2.1_bf16.safetensors"
    vae = tmp_path / "models" / "qwen_image_2.1_vae_diffusers.safetensors"
    dit.parent.mkdir()
    dit.write_bytes(b"real model file is never deleted by clearing a preference")
    vae.write_bytes(b"another real model file")
    monkeypatch.setattr(server, "PREFS_FILE", str(prefs_file))
    server.update_prefs({"qwen21_dit": str(dit), "qwen21_vae": str(vae)})

    server.update_prefs({"qwen21_dit": ""})

    saved = server.load_prefs()
    assert saved["qwen21_dit"] == ""
    assert saved["qwen21_vae"] == str(vae)
    assert dit.is_file() and vae.is_file()


def test_clearing_secondary_captioner_keeps_the_file(tmp_path, monkeypatch):
    prefs_file = tmp_path / "prefs.json"
    source = tmp_path / "qwen3vl_8b_bf16.safetensors"
    source.write_bytes(b"keep captioning model")
    monkeypatch.setattr(server, "PREFS_FILE", str(prefs_file))
    server.update_prefs({"captioner_secondary": str(source)})
    server.update_prefs({"captioner_secondary": ""})
    assert server.load_prefs()["captioner_secondary"] == ""
    assert source.is_file()


def test_preferences_model_path_rows_have_clear_actions():
    app = (Path(__file__).resolve().parents[1] /
           "src" / "fizgig" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    # Rows are now rendered by one reusable component, not repeated in renderPrefs().
    prefs_section = app.split("function mpRow(", 1)[1].split(
        "function mpOptionalPanel(", 1)[0]
    assert 'text: "Clear"' in prefs_section
    assert 'onclick: () => savePath("")' in prefs_section
    assert 'const clear = el("button"' in prefs_section
    assert "S.prefs[key] = next;" in prefs_section
    assert 'changes: {[key]: next}' in prefs_section
    caption_section = app.split("function captionerCard() {", 1)[1].split(
        "// a folder setting the two apps share", 1)[0]
    assert 'onclick: () => save("")' in caption_section
    assert "changes: {captioner_secondary: next}" in caption_section
