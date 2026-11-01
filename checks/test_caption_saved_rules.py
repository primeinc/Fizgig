"""Folder-scoped caption corrections shared by Captions and loss-watch.

CPU-only except importing the existing lightweight worker. Never downloads models.
"""
import json
from pathlib import Path

import pytest

from fizgig.caption_replace import (apply_recaption_rules, compile_saved_rules,
                                     parse_recaption_rules, normalized_saved_rules)
from fizgig.web import caption_rule_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(caption_rule_store, "user_file",
                        lambda name: str(tmp_path / "fizgig_data" / name))
    a, b = tmp_path / "Hannah", tmp_path / "Amber"
    a.mkdir(); b.mkdir()
    return a, b


def test_rules_are_saved_by_folder_not_model_family(store):
    hannah, amber = store
    rules = [{"find": "boy", "replace": "girl", "whole_word": True, "enabled": True}]
    assert caption_rule_store.load(str(hannah)) == {"enabled": False, "rules": []}
    caption_rule_store.save(str(hannah), True, rules)
    assert caption_rule_store.load(str(hannah)) == {"enabled": True, "rules": rules}
    assert caption_rule_store.load(str(amber)) == {"enabled": False, "rules": []}
    # New process / new worker sees the same file on disk.
    assert caption_rule_store.load(str(hannah))["rules"][0]["replace"] == "girl"


def test_rule_order_match_mode_and_disabled_flags(store):
    a, _ = store
    source = [
        {"find": "boy", "replace": "girl", "whole_word": True, "enabled": True},
        {"find": "girl", "replace": "woman", "whole_word": True, "enabled": False},
        {"find": "green", "replace": "blue", "whole_word": False, "enabled": True},
    ]
    cfg = caption_rule_store.save(str(a), True, source)
    result = apply_recaption_rules("A BOY and a boyfriend in a greenhouse",
                                  compile_saved_rules(cfg["rules"]))
    assert result == "A girl and a boyfriend in a bluehouse"


def test_legacy_training_rules_and_saved_folder_rules_combine():
    bundle = {"saved_rules": [{"find": "boy", "replace": "girl",
                               "whole_word": True, "enabled": True}],
              "legacy": "male => female"}
    rules = parse_recaption_rules(json.dumps(bundle))
    assert apply_recaption_rules("A BOY wearing male clothing", rules) == (
        "A girl wearing female clothing")


@pytest.mark.parametrize("bad", [
    [{"find": "", "replace": "girl"}],
    [{"find": "boy", "replace": 2}],
    [{"find": "boy", "replace": "girl", "whole_word": "sometimes"}],
    [{"find": "boy", "replace": "girl"}] * 31,
])
def test_invalid_rule_configuration_is_rejected(bad):
    with pytest.raises(ValueError):
        normalized_saved_rules(bad)


def test_worker_applies_rules_before_new_caption_write(tmp_path, monkeypatch):
    from fizgig.scripts import batch_caption
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"fake image; worker captioner is stubbed")
    listing = tmp_path / "images.txt"
    listing.write_text(str(photo) + "\n")
    instr = tmp_path / "instruction.txt"
    instr.write_text("Describe this image")
    monkeypatch.setattr(batch_caption, "_caption_qwen",
                        lambda *_args, **_kwargs: "A BOY stands near a boyfriend.")
    state = batch_caption.CaptionState("qwen", "cpu", object(), False)
    job = {"list_file": str(listing), "instruction_file": str(instr), "trigger": "hannah",
           "replacement_rules": [{"find": "boy", "replace": "girl",
                                  "whole_word": True, "enabled": True}]}
    assert not batch_caption._run_job(state, job)
    assert (tmp_path / "photo.txt").read_text() == "hannah, A girl stands near a boyfriend."


def test_worker_can_use_new_rules_with_cached_model_on_next_job(tmp_path, monkeypatch):
    from fizgig.scripts import batch_caption
    photo = tmp_path / "photo.jpg"; photo.write_bytes(b"fake")
    listing = tmp_path / "images.txt"; listing.write_text(str(photo) + "\n")
    instr = tmp_path / "instruction.txt"; instr.write_text("Describe")
    monkeypatch.setattr(batch_caption, "_caption_qwen", lambda *_args, **_kwargs: "a boy")
    state = batch_caption.CaptionState("qwen", "cpu", object(), False)
    job = {"list_file": str(listing), "instruction_file": str(instr)}
    batch_caption._run_job(state, job)
    assert (tmp_path / "photo.txt").read_text() == "a boy"
    job["replacement_rules"] = [{"find": "boy", "replace": "girl"}]
    batch_caption._run_job(state, job)
    assert (tmp_path / "photo.txt").read_text() == "a girl"


def test_ui_checkbox_and_saved_rules_are_wired_into_find_and_replace():
    repo = Path(__file__).resolve().parents[1]
    html = (repo / "src/fizgig/web/static/index.html").read_text(encoding="utf-8")
    js = (repo / "src/fizgig/web/static/app.js").read_text(encoding="utf-8")
    for marker in ('id="frRemember"', 'id="capRulesEnabled"', 'id="capRulesList"',
                   'id="frAddRule"'):
        assert marker in html
    assert 'if ($("frRemember").checked) await rememberCaptionRule()' in js
    assert 'function renderCaptionRules()' in js
    assert '"/api/captions/rules"' in js


def test_model_ui_limits_models_to_one_family_and_keeps_download_actions():
    repo = Path(__file__).resolve().parents[1]
    js = (repo / "src/fizgig/web/static/app.js").read_text(encoding="utf-8")
    section = js.split("// ---- Model library: one family at a time", 1)[1].split(
        "// Search libraries on the Fizgig computer", 1)[0]
    assert 'host.append(mpToolbar(families, selected), mpFamilyCard(selected))' in section
    assert 'text: "Download ↗"' in section
    assert 'text: "Clear"' in section
    assert 'mpHelp(sourceInfo, m)' in section
    assert 'fetchRow(f)' in section
