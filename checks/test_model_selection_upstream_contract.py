"""Upstream model build replacement contract through the fork-owned resolver.

Source: shootthesound/Fizgig@a40a60b87bb7ee2366440e9c937c8d93ac3c9c45
:: src/fizgig/scripts/fetch_models.py#fetch_weight (switch_from/replaces).

The fork may search other libraries, but it must not silently undo an upstream
build-choice or retired-filename replacement.
"""
from pathlib import Path

import pytest

from fizgig.scripts import fetch_models


@pytest.mark.parametrize("reason", ["chosen_other_build", "retired_filename"])
def test_explicit_upstream_replacement_bypasses_existing_path(
        tmp_path, monkeypatch, reason):
    old = tmp_path / "old-model.safetensors"
    new = tmp_path / "new-model.safetensors"
    old.write_bytes(b"old still available")
    new.write_bytes(b"new build already in local model cache")
    weight = fetch_models.Weight(
        "example_dit", "ExampleOrg/Model", "new-model.safetensors",
        0.1, "New model", local_name="new-model.safetensors",
        replaces=("old-model.safetensors",) if reason == "retired_filename" else (),
    )
    if reason == "chosen_other_build":
        weight.switch_from = "old-model.safetensors"

    prefs = {"example_dit": str(old)}
    calls = []
    def local_resolver(_weight, *_args, **_kwargs):
        calls.append(_weight.pref_key)
        return str(new)
    monkeypatch.setattr(fetch_models.model_sources, "resolve_weight", local_resolver)

    assert fetch_models.find_weight(weight, str(tmp_path), prefs) == str(new)
    assert fetch_models.fetch_weight(weight, str(tmp_path), prefs, log=lambda _: None)
    assert calls == ["example_dit", "example_dit"]
    assert prefs["example_dit"] == str(new)
    assert old.exists(), "Changing the selected build may not delete the previous one"


def test_upstream_build_selection_declares_what_to_replace(tmp_path):
    main = fetch_models.Weight("example_dit", "Org/Main", "main.safetensors", 1.0, "Main")
    alt = fetch_models.Weight("example_dit", "Org/Alt", "alt.safetensors", 1.0, "Alt")
    main.alt = alt
    main.fetch_choice_enabled = True
    picked = fetch_models._chosen(main, {"example_dit": "alt"})
    assert picked is not main
    assert picked.filename == "alt.safetensors"
    assert picked.switch_from == "main.safetensors"
    assert fetch_models._superseded_selection(picked, str(tmp_path / "main.safetensors"))


def test_unrelated_user_selected_model_is_preserved(tmp_path, monkeypatch):
    custom = tmp_path / "custom-finetune.safetensors"
    custom.write_bytes(b"user-selected model")
    weight = fetch_models.Weight("example_dit", "Org/Model", "new.safetensors", 1.0, "New",
                                 replaces=("old.safetensors",))
    weight.switch_from = "other-build.safetensors"
    prefs = {"example_dit": str(custom)}
    monkeypatch.setattr(fetch_models, "find_weight", lambda *_args, **_kwargs: pytest.fail(
        "Resolver must not override a valid unrelated user-selected model"))
    assert fetch_models.fetch_weight(weight, str(tmp_path), prefs, log=lambda _: None)
    assert prefs["example_dit"] == str(custom)


def test_missing_user_selected_model_is_not_silently_downloaded(tmp_path, monkeypatch):
    missing = tmp_path / "intentional-custom-file.safetensors"
    weight = fetch_models.Weight("example_dit", "Org/Model", "new.safetensors", 1.0, "New")
    prefs = {"example_dit": str(missing)}
    monkeypatch.setattr(fetch_models, "find_weight", lambda *_args, **_kwargs: None)
    logs = []
    assert not fetch_models.fetch_weight(weight, str(tmp_path), prefs, log=logs.append)
    assert any("[stale]" in line for line in logs)
    assert prefs["example_dit"] == str(missing)
    assert not missing.exists()


def test_negative_control_removing_replacement_metadata_preserves_old_file(
        tmp_path, monkeypatch):
    old = tmp_path / "old-model.safetensors"
    old.write_bytes(b"original")
    weight = fetch_models.Weight("example_dit", "Org/Model", "new.safetensors", 1.0, "New")
    prefs = {"example_dit": str(old)}
    monkeypatch.setattr(fetch_models, "find_weight", lambda *_args, **_kwargs: pytest.fail(
        "No replacement metadata: the original model must be kept"))
    assert not fetch_models._superseded_selection(weight, str(old))
    assert fetch_models.fetch_weight(weight, str(tmp_path), prefs, log=lambda _: None)
    assert prefs["example_dit"] == str(old)
