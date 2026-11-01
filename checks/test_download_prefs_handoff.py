"""Prove the Download models for me path shares Preferences with the UI.

No GPU or downloads. Fetch() must consult the exact server-owned preferences
document, respect its Qwen Image 2.1 path, and repair a stale path from the
already configured search roots before trying the network.
"""
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from fizgig import model_sources
from fizgig.scripts import fetch_models
from fizgig.utils import hf_public
from fizgig.web import server


def qwen_weight():
    return fetch_models.Weight(
        "qwen21_dit", "Comfy-Org/Qwen-Image-2.1",
        "diffusion_models/qwen_image_2.1_bf16.safetensors",
        1024 / 1024**3, "Qwen Image 2.1 DiT",
    )


def make_model(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"tensor": {"dtype": "U8", "shape": [512], "data_offsets": [0, 512]}}).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.truncate(1024)
    return path


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: ())
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: None)
    monkeypatch.setattr(fetch_models, "EMIT_PROGRESS", False)
    monkeypatch.setattr(hf_public, "download", lambda **_: pytest.fail("unexpected network transfer"))
    monkeypatch.setattr(fetch_models, "FAMILIES", {"qwen_image21": [qwen_weight()]})


def test_download_uses_same_preferences_file_as_web_and_skips_qwen(tmp_path):
    repo = tmp_path / "Fizgig"
    repo.mkdir()
    # Deliberately NOT at repo/user_data/prefs.json: the browser can use a
    # separate FIZGIG_USER_DATA and must pass the authoritative file to the child.
    prefs_file = tmp_path / "browser_user_data" / "prefs.json"
    prefs_file.parent.mkdir()
    local = make_model(repo / "user_data" / "models" / "diffusion_models"
                       / "qwen_image_2.1_bf16.safetensors")
    prefs_file.write_text(json.dumps({"qwen21_dit": str(local)}))
    logs = []
    assert fetch_models.fetch(["qwen_image21"], repo_dir=str(repo),
                              prefs_file=str(prefs_file), log=logs.append)
    assert any("[keep]" in msg and "qwen21_dit" in msg for msg in logs)
    assert json.loads(prefs_file.read_text())["qwen21_dit"] == str(local)
    assert not (repo / "user_data" / "prefs.json").exists()


def test_download_looks_up_stale_qwen_path_in_saved_roots_before_transfer(tmp_path):
    repo = tmp_path / "Fizgig"
    repo.mkdir()
    roots = tmp_path / "models"   # the USER-configured search paths, not guessed install folders
    actual = make_model(roots / "diffusion_models" / "qwen_image_2.1_bf16.safetensors")
    stale = tmp_path / "old" / "qwen_image_2.1_bf16.safetensors"
    prefs_file = tmp_path / "prefs.json"
    prefs_file.write_text(json.dumps({
        "qwen21_dit": str(stale), "model_search_roots": [str(roots)]
    }))
    logs = []
    assert fetch_models.fetch(["qwen_image21"], repo_dir=str(repo),
                              prefs_file=str(prefs_file), log=logs.append)
    assert json.loads(prefs_file.read_text())["qwen21_dit"] == str(actual)
    assert any("[have]" in msg for msg in logs)


def test_missing_saved_qwen_path_never_triggers_silent_13gb_download(tmp_path):
    repo = tmp_path / "Fizgig"
    repo.mkdir()
    missing = tmp_path / "old" / "qwen_image_2.1_bf16.safetensors"
    prefs_file = tmp_path / "prefs.json"
    prefs_file.write_text(json.dumps({
        "qwen21_dit": str(missing), "model_search_roots": [str(tmp_path / "unrelated")]
    }))
    logs = []
    assert not fetch_models.fetch(["qwen_image21"], repo_dir=str(repo),
                                  prefs_file=str(prefs_file), log=logs.append)
    assert any("[stale]" in msg and "qwen21_dit" in msg for msg in logs)
    assert json.loads(prefs_file.read_text())["qwen21_dit"] == str(missing)


def test_web_download_passes_authoritative_preferences_file_to_subprocess(tmp_path, monkeypatch):
    prefs_file = tmp_path / "user_data" / "prefs.json"
    prefs_file.parent.mkdir()
    prefs_file.write_text("{}")
    monkeypatch.setattr(server, "PREFS_FILE", str(prefs_file))
    local = make_model(tmp_path / "models" / "qwen_image_2.1_bf16.safetensors")
    monkeypatch.setattr(server, "_files_family", lambda _: SimpleNamespace(
        key="qwen_image21", model_files=(SimpleNamespace(pref_key="qwen21_dit", fetch_choice=None),),
        fetch_optional_label=""))
    calls = []
    class FakeProcess:
        def poll(self):
            return None
    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kwargs:
                        (calls.append(cmd) or FakeProcess()))
    monkeypatch.setattr(server.threading, "Thread", FakeThread)
    with server._lock:
        saved = dict(server._fetch)
    try:
        with server._lock:
            server._fetch["proc"] = None
        server._start_fetch("qwen_image21", model_paths={"qwen21_dit": str(local)})
        cmd = calls[0]
        assert cmd[cmd.index("--prefs-file") + 1] == str(prefs_file)
        assert cmd[cmd.index("--models-dir") + 1] == server.user_data.MODELS_DIR
        assert json.loads(prefs_file.read_text())["qwen21_dit"] == str(local)
    finally:
        with server._lock:
            server._fetch.clear()
            server._fetch.update(saved)
