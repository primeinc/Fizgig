"""Shared model destination layout, using tiny fake model files rather than downloads."""
import json
import os
import struct
from pathlib import Path

import pytest

from fizgig import model_destinations, model_sources
from fizgig.scripts import fetch_models


def w(path, key="demo", repo="owner/repo", filename=None):
    return fetch_models.Weight(key, repo, path, 1024 / 1024**3, "test",
                               local_name=filename)


def safetensor(path, size=1024):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = json.dumps({"tensor": {"dtype": "U8", "shape": [512], "data_offsets": [0, 512]}}).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        f.truncate(size)
    return path


@pytest.mark.parametrize(("layout", "source_path", "key", "folder"), [
    ("comfyui", "diffusion_models/moss_base.safetensors", "moss_dit", "diffusion_models"),
    ("comfyui", "text_encoders/qwen.safetensors", "moss_text_encoder", "text_encoders"),
    ("comfyui", "vae/flux2.safetensors", "moss_vae", "vae"),
    ("comfyui", "8-steps.safetensors", "moss_turbo_lora", "loras"),
    ("swarmui", "diffusion_models/moss_base.safetensors", "moss_dit", "diffusion_models"),
    ("swarmui", "text_encoders/qwen.safetensors", "moss_text_encoder", "text_encoders"),
    ("swarmui", "vae/flux2.safetensors", "moss_vae", "VAE"),
    ("swarmui", "8-steps.safetensors", "moss_turbo_lora", "Lora"),
    ("swarmui", "training_adapter.safetensors", "moss_training_adapter", "fizgig_adapters"),
])
def test_destination_uses_selected_application_folder_names(tmp_path, layout, source_path, key, folder):
    app = tmp_path / ("ComfyUI" if layout == "comfyui" else "SwarmUI")
    root = app / ("models" if layout == "comfyui" else "Models")
    root.mkdir(parents=True)
    weight = w(source_path, key=key)
    prefs = {"model_download_layout": layout, "model_download_root": str(app)}
    parent, final = model_destinations.destination(weight, tmp_path / "default", prefs)
    assert Path(parent) == root / folder
    assert Path(final) == root / folder / weight.filename
    assert not (root / folder).exists()  # planning creates no directories


def test_custom_swarmui_model_root_is_used_as_given(tmp_path):
    root = tmp_path / "custom_shared_library"
    root.mkdir()
    parent, final = model_destinations.destination(
        w("vae/test.safetensors", "test_vae"), tmp_path / "own",
        {"model_download_layout": "swarmui", "model_download_root": str(root)})
    assert Path(parent) == root / "VAE"
    assert Path(final) == root / "VAE" / "test.safetensors"


def test_default_fizgig_directory_remains_flat(tmp_path):
    weight = w("vae/diffusion_pytorch_model.safetensors")
    parent, final = model_destinations.destination(weight, tmp_path / "own", {})
    assert Path(parent) == tmp_path / "own"
    assert Path(final) == tmp_path / "own" / weight.filename


def test_generic_diffusers_file_keeps_repository_identity_in_shared_store(tmp_path):
    root = tmp_path / "ComfyUI" / "models"
    root.mkdir(parents=True)
    weight = w("vae/diffusion_pytorch_model.safetensors", "qwen21_vae", repo="Qwen/Qwen-Image-2.1")
    parent, final = model_destinations.destination(weight, tmp_path / "own",
        {"model_download_layout": "comfyui", "model_download_root": str(root)})
    assert Path(final) == root / "vae" / "Qwen" / "Qwen-Image-2.1" / "vae" / weight.filename


def test_selected_missing_library_refuses_to_create_application(tmp_path):
    missing = tmp_path / "ComfyUI"
    with pytest.raises(ValueError, match="does not exist"):
        model_destinations.destination(w("vae/a.safetensors"), tmp_path / "own",
            {"model_download_layout": "comfyui", "model_download_root": str(missing)})
    assert not missing.exists()


def test_existing_explicit_path_still_wins_over_download_target(tmp_path):
    shared = tmp_path / "ComfyUI" / "models"
    shared.mkdir(parents=True)
    manual = tmp_path / "custom.safetensors"
    manual.write_bytes(b"personal model")
    weight = w("vae/moss_vae.safetensors", "moss_vae")
    prefs = {"model_download_layout": "comfyui", "model_download_root": str(shared),
             weight.pref_key: str(manual)}
    assert fetch_models.fetch_weight(weight, str(tmp_path / "own"), prefs, log=lambda _: None)
    assert prefs[weight.pref_key] == str(manual)
    assert not (shared / "vae").exists()


def test_download_to_comfy_layout_reuses_saved_file_on_next_attempt(tmp_path, monkeypatch):
    root = tmp_path / "ComfyUI" / "models"
    root.mkdir(parents=True)
    weight = w("diffusion_models/model.safetensors")
    prefs = {"model_download_layout": "comfyui", "model_download_root": str(root)}
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: ())
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: None)
    from fizgig.utils import hf_public
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(safetensor(Path(kwargs["local_dir"]) / kwargs["filename"]))
    monkeypatch.setattr(hf_public, "download", download)
    assert fetch_models.fetch_weight(weight, str(tmp_path / "own"), prefs, log=lambda _: None)
    expected = root / "diffusion_models" / weight.filename
    assert prefs[weight.pref_key] == str(expected)
    assert expected.is_file()
    assert not (root / "diffusion_models" / "diffusion_models" / weight.filename).exists()
    prefs.pop(weight.pref_key)
    assert fetch_models.fetch_weight(weight, str(tmp_path / "own"), prefs, log=lambda _: None)
    assert prefs[weight.pref_key] == str(expected)
    assert len(calls) == 1


def test_bad_existing_model_is_not_treated_as_downloaded(tmp_path, monkeypatch):
    root = tmp_path / "SwarmUI" / "Models"
    root.mkdir(parents=True)
    weight = w("vae/model.safetensors", "model_vae")
    dest = root / "VAE" / weight.filename
    dest.parent.mkdir()
    dest.write_text("bad bytes")
    prefs = {"model_download_layout": "swarmui", "model_download_root": str(root)}
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: ())
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: None)
    from fizgig.utils import hf_public
    def download(**kwargs):
        return str(safetensor(Path(kwargs["local_dir"]) / kwargs["filename"]))
    monkeypatch.setattr(hf_public, "download", download)
    assert fetch_models.fetch_weight(weight, str(tmp_path / "own"), prefs, log=lambda _: None)
    assert fetch_models._valid_safetensors(str(dest), 819)
    assert prefs["model_vae"] == str(dest)
