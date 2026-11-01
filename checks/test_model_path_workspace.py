"""Code contracts for schema-driven Model Paths UI.

No fake hard-coded model catalog and no destructive replacement of the
per-family downloader. Actual DOM is checked by the JavaScript parse stage.
"""
from pathlib import Path


def app_source():
    return (Path(__file__).resolve().parents[1] /
            "src/fizgig/web/static/app.js").read_text(encoding="utf-8")


def test_workspace_shares_family_descriptions_and_original_download_controls():
    source = app_source()
    section = source.split("// ---- Preferences: compact model paths", 1)[1].split(
        "// Search libraries on the Fizgig computer", 1)[0]
    assert section.count("function mpRow(") == 1
    assert section.count("function renderPrefs()") == 1
    assert "mpModels(f)" in section
    assert "S.schema.families.filter" in section
    assert "fetchRow(f)" in section
    assert "modelFiles =" not in section
    assert "mpRefresh()" in section


def test_all_model_rows_have_keyboard_accessible_details_and_clear():
    section = app_source().split("function mpRow(", 1)[1].split("function mpOptionalPanel(", 1)[0]
    assert "mpHelp(sourceNote, m, alternative)" in section
    assert '"aria-label": "Read model details"' in app_source()
    assert 'onclick: () => savePath("")' in section
    assert "changes: {[key]: next}" in section
    assert '"https://huggingface.co/"' in section


def test_storage_metrics_are_bound_to_server_inventory_not_guessing_from_textboxes():
    section = app_source().split("// ---- Preferences: compact model paths", 1)[1].split(
        "// Search libraries on the Fizgig computer", 1)[0]
    assert "S.modelInventory?.files" in section
    assert "S.modelInventory?.summary" in section
    assert "st.bytes" in section
    assert "remaining_required_bytes" in section
    assert "mpSize(st.total)" in app_source()
    assert "st.total - st.done" in app_source()
