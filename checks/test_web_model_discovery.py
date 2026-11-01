"""Web-side model discovery: exercise the real worker and preference persistence, CPU only."""
import json
import struct

import pytest

from fizgig import model_sources
from fizgig.scripts import fetch_models
from fizgig.web import server


def tiny_weight(name="example.safetensors"):
    return fetch_models.Weight("test_model", "owner/repo",
                               "diffusion_models/" + name, 1024 / 1024**3, "test")


def tiny_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = json.dumps({"tensor": {"dtype": "U8", "shape": [512], "data_offsets": [0, 512]}}).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        f.truncate(1024)
    return path


@pytest.fixture
def web_search(tmp_path, monkeypatch):
    root = tmp_path / "models_lib"
    root.mkdir()
    local = tmp_path / "models"
    local.mkdir()
    prefs_file = tmp_path / "prefs.json"
    prefs_file.write_text(json.dumps({"model_search_roots": [str(root)]}), encoding="utf-8")
    monkeypatch.setattr(server, "PREFS_FILE", str(prefs_file))
    monkeypatch.setattr(server.user_data, "MODELS_DIR", str(local))
    monkeypatch.setattr(server, "training_families", lambda: [])
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: ())
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: None)
    w = tiny_weight()
    monkeypatch.setattr(fetch_models, "FAMILIES", {"demo": [w]})

    class SynchronousThread:
        def __init__(self, target, **_):
            self.target = target
        def start(self):
            self.target()

    monkeypatch.setattr(server.threading, "Thread", SynchronousThread)
    with server._lock:
        server._model_search["state"] = "idle"
        server._fetch["proc"] = None
    return root, prefs_file, w


def test_web_scan_finds_weight_and_persists_pref(web_search):
    root, prefs_file, w = web_search
    expected = tiny_file(root / "ComfyUI" / "models" / w.filename)
    server._start_model_search("demo")
    status = server._model_search_status()
    assert status["state"] == "done"
    assert (status["found"], status["kept"], status["missing"], status["ambiguous"]) == (1, 0, 0, 0)
    assert json.loads(prefs_file.read_text())[w.pref_key] == str(expected)


def test_web_scan_does_not_override_existing_user_choice(web_search):
    root, prefs_file, w = web_search
    manual = root / "manually-selected.safetensors"
    manual.write_bytes(b"my model")
    pref = json.loads(prefs_file.read_text())
    pref[w.pref_key] = str(manual)
    prefs_file.write_text(json.dumps(pref))
    tiny_file(root / w.filename)
    server._start_model_search("demo")
    assert server._model_search_status()["kept"] == 1
    assert json.loads(prefs_file.read_text())[w.pref_key] == str(manual)


def test_web_scan_does_not_overwrite_user_edit_during_search(web_search, monkeypatch):
    root, prefs_file, w = web_search
    manual = root / "manual.safetensors"
    result = tiny_file(root / w.filename)
    def race(_w, _models_dir, _prefs, scan_cache=None):
        server.update_prefs({w.pref_key: str(manual)})
        return str(result)
    monkeypatch.setattr(fetch_models, "find_weight", race)
    server._start_model_search("demo")
    assert server._model_search_status()["state"] == "done"
    assert json.loads(prefs_file.read_text())[w.pref_key] == str(manual)


def test_ambiguous_web_scan_does_not_write_pref(web_search):
    root, prefs_file, w = web_search
    tiny_file(root / "one" / w.filename)
    tiny_file(root / "two" / w.filename)
    server._start_model_search("demo")
    status = server._model_search_status()
    assert status["state"] == "done"
    assert status["ambiguous"] == 1
    assert w.pref_key not in json.loads(prefs_file.read_text())
