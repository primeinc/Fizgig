"""Download models for me must never replace an existing configured model path.

This runs the REAL fetch() entrypoint used by the button; the network function
fails the test if the downloader tries to fetch a weight that Preferences already
points at. No GPU, no real model downloads.
"""
import json

import pytest

from fizgig import model_sources
from fizgig.scripts import fetch_models
from fizgig.utils import hf_public


@pytest.fixture(autouse=True)
def no_remote_downloads(monkeypatch):
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: ())
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: None)
    monkeypatch.setattr(fetch_models, "EMIT_PROGRESS", False)
    monkeypatch.setattr(hf_public, "download",
                        lambda **_: pytest.fail("Download models for me tried to redownload an existing configured file"))


def _prefs(repo_dir, key, path):
    settings = repo_dir / "user_data"
    settings.mkdir(parents=True, exist_ok=True)
    (settings / "prefs.json").write_text(json.dumps({key: str(path)}), encoding="utf-8")


def test_moss_replaced_filename_is_still_an_explicit_valid_model(tmp_path, monkeypatch):
    repo = tmp_path / "Fizgig"
    old = tmp_path / "shared" / "kromoss_bf16.safetensors"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"an existing user-selected model")
    w = fetch_models.Weight("moss_dit", "Comfy-Org/MOSS-Image",
                            "diffusion_models/moss_image_sft_bf16.safetensors",
                            1 / 1024 ** 3, "MOSS Image", replaces=("kromoss_bf16.safetensors",))
    _prefs(repo, w.pref_key, old)
    monkeypatch.setattr(fetch_models, "FAMILIES", {"moss_image": [w]})
    logs = []
    assert fetch_models.fetch(["moss_image"], repo_dir=str(repo), log=logs.append)
    assert fetch_models._load_prefs(str(repo / "user_data" / "prefs.json"))[w.pref_key] == str(old)
    assert any("[ok]" in line or "[keep]" in line for line in logs)


def test_download_build_choice_does_not_override_existing_path(tmp_path, monkeypatch):
    repo = tmp_path / "Fizgig"
    old = tmp_path / "shared" / "kromoss_int8_convrot.safetensors"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"existing intentionally selected build")
    main = fetch_models.Weight("moss_dit", "Comfy-Org/MOSS-Image",
                               "diffusion_models/moss_image_base_bf16.safetensors",
                               1 / 1024 ** 3, "main")
    alternate = fetch_models.Weight("moss_dit", "Comfy-Org/MOSS-Image",
                                    "diffusion_models/kromoss_int8_convrot.safetensors",
                                    1 / 1024 ** 3, "alternative")
    main.alt = alternate
    _prefs(repo, "moss_dit", old)
    monkeypatch.setattr(fetch_models, "FAMILIES", {"moss_image": [main]})
    assert fetch_models.fetch(["moss_image"], choices={"moss_dit": "main"}, repo_dir=str(repo))
    assert fetch_models._load_prefs(str(repo / "user_data" / "prefs.json"))["moss_dit"] == str(old)


def test_relative_configured_path_resolves_relative_to_fizgig_repository(tmp_path, monkeypatch):
    repo = tmp_path / "Fizgig"
    path = repo / "user_data" / "models" / "my_model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing user model")
    relative = "user_data/models/my_model.safetensors"
    _prefs(repo, "moss_dit", relative)
    w = fetch_models.Weight("moss_dit", "Comfy-Org/MOSS-Image",
                            "diffusion_models/my_model.safetensors",
                            1 / 1024 ** 3, "MOSS Image")
    monkeypatch.setattr(fetch_models, "FAMILIES", {"moss_image": [w]})
    assert fetch_models.fetch(["moss_image"], repo_dir=str(repo))
    assert fetch_models._load_prefs(str(repo / "user_data" / "prefs.json"))["moss_dit"] == relative
