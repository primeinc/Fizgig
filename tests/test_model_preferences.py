import json
import tkinter as tk
from unittest.mock import Mock

import pytest

import lora_trainer_gui as gui
from fizgig.scripts import fetch_models


@pytest.fixture
def preferences(tmp_path, monkeypatch):
    try:
        master = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk UI tests need a display (use xvfb-run on Linux)")
    master.withdraw()
    monkeypatch.setattr(gui, "PREFS_FILE", str(tmp_path / "prefs.json"))
    monkeypatch.delenv("FIZGIG_NO_PERSIST", raising=False)
    app = gui.LoRATrainerGUI.__new__(gui.LoRATrainerGUI)
    app.master = master
    app.prefs = {"model_search_roots": []}
    app.prefs_vars = {}
    app._create_model_search_roots(master)
    yield app
    master.destroy()


@pytest.mark.parametrize("count", [0, 1, 3])
def test_gui_loads_and_persists_root_lists(preferences, tmp_path, count):
    roots = [str(tmp_path / f"root-{index}") for index in range(count)]
    preferences.prefs["model_search_roots"] = roots
    gui.save_prefs(preferences.prefs)

    loaded = gui.load_prefs()

    assert loaded["model_search_roots"] == roots


def test_add_root_preserves_application_selection(preferences, tmp_path, monkeypatch):
    selected = tmp_path / "ComfyUI"
    (selected / "models").mkdir(parents=True)
    monkeypatch.setattr(gui.filedialog, "askdirectory", lambda **_: str(selected))

    preferences._edit_model_search_roots("add")

    assert preferences.prefs["model_search_roots"] == [str(selected)]
    assert json.loads((tmp_path / "prefs.json").read_text())["model_search_roots"] == [
        str(selected)
    ]
    assert preferences._model_roots_list.get(0) == str(selected)


@pytest.mark.parametrize(("action", "index"), [("up", 1), ("down", 0)])
def test_reorder_roots_persists(preferences, tmp_path, action, index):
    roots = [str(tmp_path / "first"), str(tmp_path / "second")]
    preferences._model_roots_list.insert(tk.END, *roots)
    preferences._model_roots_list.selection_set(index)

    preferences._edit_model_search_roots(action)

    assert gui.load_prefs()["model_search_roots"] == roots[::-1]
    assert list(preferences._model_roots_list.get(0, tk.END)) == roots[::-1]


def test_remove_last_root_persists_empty_list(preferences, tmp_path):
    preferences._model_roots_list.insert(tk.END, str(tmp_path))
    preferences._model_roots_list.selection_set(0)

    preferences._edit_model_search_roots("remove")

    assert gui.load_prefs()["model_search_roots"] == []
    assert preferences._model_roots_list.size() == 0


def test_cancel_add_does_not_change_roots(preferences, monkeypatch):
    monkeypatch.setattr(gui.filedialog, "askdirectory", lambda **_: "")

    preferences._edit_model_search_roots("add")

    assert preferences.prefs["model_search_roots"] == []
    assert preferences._model_roots_list.size() == 0


def test_reset_clears_roots(preferences, tmp_path, monkeypatch):
    preferences.prefs["model_search_roots"] = [str(tmp_path)]
    preferences._model_roots_list.insert(tk.END, str(tmp_path))
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *_: True)

    preferences._reset_prefs()

    assert gui.load_prefs()["model_search_roots"] == []
    assert preferences._model_roots_list.size() == 0


@pytest.mark.parametrize("resolved", [True, False])
def test_klein_probes_existing_weights_before_token_prompt(
    preferences, monkeypatch, resolved
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    weight = fetch_models.Weight(
        "base_dit", "owner/repo", "model.safetensors", 1, "", gated=True
    )
    monkeypatch.setattr(fetch_models, "FAMILIES", {"klein": [weight]})
    monkeypatch.setattr(
        fetch_models, "find_weight", lambda *_: "/local/model" if resolved else None
    )
    thread = Mock()
    monkeypatch.setattr(gui.threading, "Thread", thread)
    preferences._ask_hf_token = Mock()

    preferences._start_fetch_models("klein")

    preferences._ask_hf_token.assert_not_called()
    assert preferences._fetch_running is True
    thread.return_value.start.assert_called_once()
    resume = Mock()
    preferences._start_fetch_models = resume
    thread.call_args.kwargs["target"]()
    preferences.master.update()
    resume.assert_called_once_with("klein", _needs_token=not resolved)


def test_unresolved_klein_keeps_token_cancel_flow(preferences, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    preferences._ask_hf_token = Mock(return_value="")
    thread = Mock()
    monkeypatch.setattr(gui.threading, "Thread", thread)

    preferences._start_fetch_models("klein", _needs_token=True)

    preferences._ask_hf_token.assert_called_once()
    thread.assert_not_called()
