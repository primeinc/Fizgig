"""Intentional downstream model-selection policy, compared with upstream semantics.

Upstream: shootthesound/Fizgig@a40a60b87bb7ee2366440e9c937c8d93ac3c9c45
:: src/fizgig/scripts/fetch_models.py#fetch_weight uses switch_from/replaces
to supersede a configured filename. This fork deliberately diverges: valid
explicit Preferences paths have priority regardless of basename, because
the web UI passes a build-choice value even without a user-triggered change.

To replace a model, the user clears the model's Preferences row, then chooses
a build and downloads. Missing saved files are not silently redownloaded.
The next upstream sync must preserve this documented exception.
"""
import pytest

from fizgig.scripts import fetch_models


@pytest.mark.parametrize("reason", ["build_choice", "retired_filename"])
def test_existing_configured_file_wins_over_upstream_replacement_metadata(
        tmp_path, monkeypatch, reason):
    existing = tmp_path / "old-model.safetensors"
    existing.write_bytes(b"user-selected model")
    model = fetch_models.Weight(
        "example_dit", "ExampleOrg/Model", "new-model.safetensors", 0.1,
        "New model",
        replaces=("old-model.safetensors",) if reason == "retired_filename" else (),
    )
    if reason == "build_choice":
        model.switch_from = "old-model.safetensors"
    prefs = {"example_dit": str(existing)}

    def fail_on_search(*_args, **_kwargs):
        pytest.fail("Existing configured model path must not trigger discovery")
    monkeypatch.setattr(fetch_models, "find_weight", fail_on_search)
    assert fetch_models.fetch_weight(model, str(tmp_path), prefs, log=lambda _: None)
    assert prefs["example_dit"] == str(existing)
    assert existing.read_bytes() == b"user-selected model"


def test_build_choice_still_selects_requested_catalog_variant(tmp_path):
    main = fetch_models.Weight("example_dit", "Org/Main", "main.safetensors", 1.0, "Main")
    alt = fetch_models.Weight("example_dit", "Org/Alt", "alt.safetensors", 1.0, "Alt")
    main.alt = alt
    main.fetch_choice_enabled = True
    selected = fetch_models._chosen(main, {"example_dit": "alt"})
    assert selected.filename == "alt.safetensors"
    assert selected.switch_from == "main.safetensors"


def test_clear_path_permits_new_build_and_invokes_resolver(tmp_path, monkeypatch):
    old = tmp_path / "main.safetensors"
    newer = tmp_path / "alt.safetensors"
    old.write_bytes(b"existing old file")
    newer.write_bytes(b"candidate already cached")
    selected = fetch_models.Weight("example_dit", "Org/Alt", "alt.safetensors", 1.0, "Alt")
    selected.switch_from = "main.safetensors"
    prefs = {"example_dit": ""}
    used = []

    def candidate(_w, *_args, **_kwargs):
        used.append(_w.filename)
        return str(newer)

    monkeypatch.setattr(fetch_models, "find_weight", candidate)
    assert fetch_models.fetch_weight(selected, str(tmp_path), prefs, log=lambda _: None)
    assert prefs["example_dit"] == str(newer)
    assert used == ["alt.safetensors"]
    assert old.exists()


def test_stale_explicit_path_never_authorizes_implicit_download(tmp_path, monkeypatch):
    missing = tmp_path / "old-model.safetensors"
    model = fetch_models.Weight("example_dit", "Org/Model", "new.safetensors",
                                1.0, "New", replaces=("old-model.safetensors",))
    model.switch_from = "old-model.safetensors"
    prefs = {"example_dit": str(missing)}
    monkeypatch.setattr(fetch_models, "find_weight", lambda *_a, **_kw: None)
    logs = []
    assert not fetch_models.fetch_weight(model, str(tmp_path), prefs, log=logs.append)
    assert any("[stale]" in line for line in logs)
    assert prefs["example_dit"] == str(missing)


def test_negative_control_removing_path_guard_changes_fetch_behavior(tmp_path, monkeypatch):
    valid = tmp_path / "old.safetensors"
    valid.write_bytes(b"keep this")
    model = fetch_models.Weight("example_dit", "Org/Model", "new.safetensors", 1.0, "New")
    prefs = {"example_dit": str(valid)}
    assert fetch_models.fetch_weight(model, str(tmp_path), prefs, log=lambda _: None)
    # A cleared selection MUST reach discovery; the same test would fail if a
    # code change permanently suppressed the resolver.
    prefs["example_dit"] = ""
    calls = []
    def missing_candidate(*_a, **_kw):
        calls.append("searched")
        return None
    monkeypatch.setattr(fetch_models, "find_weight", missing_candidate)
    assert fetch_models.fetch_weight(model, str(tmp_path), prefs, log=lambda _: None, dry_run=True)
    assert calls == ["searched"]
