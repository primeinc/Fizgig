"""Model discovery: real synthetic filesystem trees, no GPU, network or model downloads."""
import json
import os
import struct
from pathlib import Path

import pytest

from fizgig import model_sources
from fizgig.scripts import fetch_models

# Save before the offline fixture patches it for ordinary resolution tests.
real_automatic_roots = model_sources.automatic_roots


def weight(name="my_model.safetensors", pref="demo", repo="owner/repo", path=None, **kwargs):
    return fetch_models.Weight(pref, repo, path or "diffusion_models/" + name,
                               1024 / 1024**3, "tiny test weight", **kwargs)


def safetensor(path, size=1024):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"tensor": {"dtype": "U8", "shape": [512], "data_offsets": [0, 512]}}).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.truncate(size)
    return path


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: None)
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: ())
    from fizgig.utils import hf_public
    monkeypatch.setattr(hf_public, "download", lambda **_: pytest.fail("unexpected network download"))


def resolve(w, roots, own):
    return model_sources.resolve_weight(w, roots, own, fetch_models._valid_safetensors, 819)


def test_explicit_community_choice_is_never_replaced(tmp_path):
    w = weight(gated=True)
    custom = tmp_path / "personal.safetensors"
    custom.write_bytes(b"not the same as the canonical model")
    safetensor(tmp_path / "ComfyUI" / "models" / w.filename)
    prefs = {"model_search_roots": [str(tmp_path / "ComfyUI")], w.pref_key: str(custom)}
    assert fetch_models.fetch_weight(w, str(tmp_path / "downloads"), prefs, log=lambda _: None)
    assert prefs[w.pref_key] == str(custom)


def test_comfy_root_reuses_a_file_without_downloading(tmp_path):
    w = weight(gated=True)
    path = safetensor(tmp_path / "ComfyUI" / "models" / "diffusion_models" / w.filename)
    prefs = {"model_search_roots": [str(tmp_path / "ComfyUI")]}
    assert fetch_models.fetch_weight(w, str(tmp_path / "downloads"), prefs, log=lambda _: None)
    assert prefs[w.pref_key] == str(path)
    assert not (tmp_path / "downloads").exists()


@pytest.mark.parametrize(("app", "marker", "subtree"), [
    ("ComfyUI", None, "models"),
    ("other", "folder_paths.py", "models"),
    ("SwarmUI", None, "Models"),
    ("stable-diffusion-webui-forge", None, "models"),
    ("Fooocus", None, "models"),
    ("a_library", ".sm-portable", "Models"),
])
def test_recognized_app_folder_is_narrowed(tmp_path, app, marker, subtree):
    root = tmp_path / app
    (root / subtree).mkdir(parents=True)
    if marker:
        mark = root / marker
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.touch()
    assert model_sources.effective_roots(str(root)) == (root / subtree,)


def test_roots_have_priority_over_cache_and_own_models(tmp_path, monkeypatch):
    w = weight()
    first = safetensor(tmp_path / "first" / w.filename)
    safetensor(tmp_path / "second" / w.filename)
    cache = safetensor(tmp_path / "cache" / w.filename)
    safetensor(tmp_path / "downloads" / w.filename)
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: str(cache))
    assert resolve(w, [str(first.parent), str(tmp_path / "second")], tmp_path / "downloads") == str(first)


def test_ambiguous_first_root_blocks_lower_precedence_and_download(tmp_path):
    w = weight()
    root = tmp_path / "first"
    safetensor(root / "a" / w.filename)
    safetensor(root / "b" / w.filename)
    safetensor(tmp_path / "second" / w.filename)
    prefs = {"model_search_roots": [str(root), str(tmp_path / "second")]}
    logs = []
    assert not fetch_models.fetch_weight(w, str(tmp_path / "downloads"), prefs, log=logs.append)
    assert w.pref_key not in prefs
    assert any("[ambiguous]" in line for line in logs)


def test_generic_diffusers_filename_needs_declared_repository_identity(tmp_path):
    w = weight("diffusion_pytorch_model.safetensors", repo="Qwen/Qwen-Image-2.1",
               path="vae/diffusion_pytorch_model.safetensors")
    root = tmp_path / "shared"
    safetensor(root / "different" / "vae" / w.filename)
    assert resolve(w, [str(root)], tmp_path / "downloads") is None
    expected = safetensor(root / "Qwen" / "Qwen-Image-2.1" / "vae" / w.filename)
    assert resolve(w, [str(root)], tmp_path / "downloads") == str(expected)


def test_invalid_header_or_truncated_model_never_matches(tmp_path):
    w = weight()
    found = safetensor(tmp_path / "ComfyUI" / "models" / w.filename)
    found.write_bytes(b"invalid header".ljust(1024, b" "))
    assert resolve(w, [str(found.parent)], tmp_path / "downloads") is None
    safetensor(found, 300)
    assert resolve(w, [str(found.parent)], tmp_path / "downloads") is None


def test_hugging_face_cache_hit_reuses_path(tmp_path, monkeypatch):
    w = weight()
    cached = safetensor(tmp_path / "hub" / w.filename)
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: str(cached))
    prefs = {}
    assert fetch_models.fetch_weight(w, str(tmp_path / "downloads"), prefs, log=lambda _: None)
    assert prefs[w.pref_key] == str(cached)


def test_stability_matrix_library_pointer(tmp_path, monkeypatch):
    root = tmp_path / "collection"
    (root / "Models").mkdir(parents=True)
    pointer = tmp_path / "cfg" / "StabilityMatrix" / "library.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"LibraryPath": str(root)}), encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "cfg"))
    assert real_automatic_roots() == (root / "Models",)


def test_model_candidates_use_only_the_canonical_fetch_catalog(monkeypatch):
    one = weight(pref="one")
    shared = weight(pref="shared")
    two = weight(pref="two", optional=True)
    monkeypatch.setattr(fetch_models, "FAMILIES", {"a": [one, shared], "b": [shared, two]})
    assert [w.pref_key for w in fetch_models.model_candidates()] == ["one", "shared", "two"]
    assert [w.pref_key for w in fetch_models.model_candidates(("b",), include_optional=False)] == ["shared"]


def test_model_build_choice_is_shared_between_discovery_and_download(monkeypatch):
    main = weight(pref="model", name="base.safetensors")
    alternate = weight(pref="model", name="alternate.safetensors")
    main.alt = alternate
    monkeypatch.setattr(fetch_models, "FAMILIES", {"a": [main]})
    selected, = fetch_models.model_candidates(("a",), choices={"model": "alt"})
    assert selected.filename == "alternate.safetensors"
    assert selected.switch_from == "base.safetensors"


def test_same_as_pref_reuses_the_users_other_model(tmp_path):
    w = weight(pref="second")
    w.same_as = "first"
    manual = tmp_path / "my-vae.safetensors"
    manual.write_bytes(b"custom variant")
    prefs = {"first": str(manual)}
    assert fetch_models.fetch_weight(w, str(tmp_path / "downloads"), prefs, log=lambda _: None)
    assert prefs["second"] == str(manual)


def test_fetch_does_not_erase_roots_edited_during_a_download(tmp_path, monkeypatch):
    prefs_file = tmp_path / "user_data" / "prefs.json"
    prefs_file.parent.mkdir(parents=True)
    fetch_models._save_prefs(str(prefs_file), {"model_search_roots": ["/initial"]})
    w = weight()
    monkeypatch.setattr(fetch_models, "FAMILIES", {"demo": [w]})

    def intervening_download(w, models_dir, prefs, **_):
        prefs[w.pref_key] = str(tmp_path / w.filename)
        fetch_models._save_prefs(str(prefs_file), {"model_search_roots": ["/changed"]})
        return True

    monkeypatch.setattr(fetch_models, "fetch_weight", intervening_download)
    assert fetch_models.fetch(["demo"], repo_dir=str(tmp_path), log=lambda _: None)
    saved = fetch_models._load_prefs(str(prefs_file))
    assert saved["model_search_roots"] == ["/changed"]
    assert saved[w.pref_key] == str(tmp_path / w.filename)


def test_scan_cache_walks_a_library_only_once_for_multiple_models(tmp_path, monkeypatch):
    one = weight(name="first_model.safetensors", pref="one")
    two = weight(name="second_model.safetensors", pref="two")
    root = tmp_path / "ComfyUI"
    safetensor(root / "models" / one.filename)
    safetensor(root / "models" / two.filename)
    prefs = {"model_search_roots": [str(root)]}
    walks = []
    real_walk = model_sources.os.walk

    def count_walk(*args, **kwargs):
        walks.append(str(args[0]))
        return real_walk(*args, **kwargs)

    monkeypatch.setattr(model_sources.os, "walk", count_walk)
    cache = {}
    assert fetch_models.find_weight(one, str(tmp_path / "downloads"), prefs, scan_cache=cache)
    assert fetch_models.find_weight(two, str(tmp_path / "downloads"), prefs, scan_cache=cache)
    assert walks == [str(root / "models")]


def test_missing_gated_weight_needs_token_only_after_local_search(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    logs = []
    w = weight(gated=True)
    assert not fetch_models.fetch_weight(w, str(tmp_path / "downloads"), {}, log=logs.append)
    assert any("[gated]" in line for line in logs)


def test_fetch_never_overwrites_manual_path_saved_while_downloading(tmp_path, monkeypatch):
    prefs_file = tmp_path / "user_data" / "prefs.json"
    prefs_file.parent.mkdir(parents=True)
    w = weight()
    monkeypatch.setattr(fetch_models, "FAMILIES", {"demo": [w]})
    fetch_models._save_prefs(str(prefs_file), {})

    def intervening_download(w, models_dir, prefs, **_):
        prefs[w.pref_key] = str(tmp_path / "downloaded.safetensors")
        fetch_models._save_prefs(str(prefs_file), {w.pref_key: str(tmp_path / "manual.safetensors")})
        return True

    monkeypatch.setattr(fetch_models, "fetch_weight", intervening_download)
    assert fetch_models.fetch(["demo"], repo_dir=str(tmp_path), log=lambda _: None)
    assert fetch_models._load_prefs(str(prefs_file))[w.pref_key] == str(tmp_path / "manual.safetensors")
