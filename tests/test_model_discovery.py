import json
import os
import struct
from pathlib import Path

import pytest

from fizgig import model_sources
from fizgig.scripts import fetch_models

automatic_roots = model_sources.automatic_roots


def make_weight(path, size=1024):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = max(1, size - 256)
    header = (
        json.dumps(
            {
                "tensor": {
                    "dtype": "U8",
                    "shape": [payload],
                    "data_offsets": [0, payload],
                }
            }
        )
        .encode()
        .ljust(size - payload - 8, b" ")
    )
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.truncate(size)
    return path


@pytest.fixture
def weight():
    return fetch_models.Weight(
        "test_weight",
        "owner/repo",
        "diffusion_models/model.safetensors",
        1024 / 1024**3,
        "Test weight",
    )


@pytest.fixture(autouse=True)
def isolate_stores(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: None)
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: ())

    def unexpected_download(**_kwargs):
        pytest.fail("A resolved weight must not download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)


def discover(weight, roots, models_dir):
    return model_sources.resolve_weight(
        weight, roots, models_dir, fetch_models._valid_safetensors, 819
    )


def test_manual_search_uses_only_selected_roots(tmp_path, weight):
    make_weight(tmp_path / "downloads" / weight.filename)

    result = model_sources.search_model_roots(
        weight, [], fetch_models._valid_safetensors, 819
    )

    assert result is None


@pytest.mark.parametrize("count", [0, 1, 3])
def test_roots_persist_through_fetch(tmp_path, monkeypatch, count):
    roots = [str(tmp_path / f"root-{index}") for index in range(count)]
    fetch_models._save_prefs(
        str(tmp_path / "prefs.json"), {"model_search_roots": roots}
    )
    monkeypatch.setattr(fetch_models, "FAMILIES", {"test": []})

    assert fetch_models.fetch(["test"], repo_dir=str(tmp_path), log=lambda _: None)

    assert (
        fetch_models._load_prefs(str(tmp_path / "prefs.json"))["model_search_roots"]
        == roots
    )


def test_arbitrary_nested_root(tmp_path, weight):
    expected = make_weight(tmp_path / "library" / "deep" / "other" / weight.filename)

    result = discover(weight, [str(tmp_path / "library")], tmp_path / "downloads")

    assert result == str(expected)


def test_fetch_preserves_search_roots_changed_while_running(tmp_path, monkeypatch):
    prefs_file = tmp_path / "prefs.json"
    fetch_models._save_prefs(
        str(prefs_file), {"model_search_roots": [str(tmp_path / "before")]}
    )
    weight = fetch_models.Weight("test_weight", "owner/repo", "model.safetensors", 1, "")
    monkeypatch.setattr(fetch_models, "FAMILIES", {"test": [weight]})

    def resolve_during_fetch(model, models_dir, prefs, **_kwargs):
        prefs[model.pref_key] = str(tmp_path / model.filename)
        fetch_models._save_prefs(
            str(prefs_file),
            {"model_search_roots": [str(tmp_path / "after")]},
        )
        return True

    monkeypatch.setattr(fetch_models, "fetch_weight", resolve_during_fetch)

    assert fetch_models.fetch(["test"], repo_dir=str(tmp_path), log=lambda _: None)

    prefs = fetch_models._load_prefs(str(prefs_file))
    assert prefs["model_search_roots"] == [str(tmp_path / "after")]
    assert prefs["test_weight"] == str(tmp_path / weight.filename)


@pytest.mark.parametrize("reverse", [False, True])
def test_ordered_roots_win_over_cache_and_download_directory(
    tmp_path, weight, monkeypatch, reverse
):
    candidates = [
        make_weight(tmp_path / name / weight.filename) for name in ("first", "second")
    ]
    cached = make_weight(tmp_path / "cache" / weight.filename)
    make_weight(tmp_path / "downloads" / weight.filename)
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: str(cached))
    if reverse:
        candidates.reverse()

    result = discover(
        weight, [str(path.parent) for path in candidates], tmp_path / "downloads"
    )

    assert result == str(candidates[0])


def test_explicit_community_path_is_preserved_without_size_check(tmp_path, weight):
    explicit = tmp_path / "community.safetensors"
    explicit.write_bytes(b"community variant")
    prefs = {weight.pref_key: str(explicit)}
    make_weight(tmp_path / "downloads" / weight.filename)

    result = fetch_models.fetch_weight(weight, str(tmp_path / "downloads"), prefs)

    assert result is True
    assert prefs[weight.pref_key] == str(explicit)


@pytest.mark.parametrize(
    ("application", "marker", "subtree"),
    [
        ("renamed-comfy", "folder_paths.py", "models"),
        ("ComfyUI", None, "models"),
        ("forge", "modules/paths_internal.py", "models"),
        ("SwarmUI", None, "Models"),
        ("Fooocus", None, "models"),
        ("library", ".sm-portable", "Models"),
    ],
)
def test_registry_normalizes_application_roots(tmp_path, application, marker, subtree):
    root = tmp_path / application
    (root / subtree).mkdir(parents=True)
    if marker:
        (root / marker).parent.mkdir(parents=True, exist_ok=True)
        (root / marker).touch()

    result = model_sources.effective_roots(str(root))

    assert result == (root / subtree,)


def test_comfy_root_search_is_narrowed(tmp_path, weight):
    root = tmp_path / "ComfyUI"
    expected = make_weight(root / "models" / "unet" / weight.filename)
    make_weight(root / "irrelevant" / weight.filename)

    result = discover(weight, [str(root)], tmp_path / "downloads")

    assert result == str(expected)


def test_portable_wrapper(tmp_path):
    models = tmp_path / "portable" / "ComfyUI" / "models"
    models.mkdir(parents=True)

    result = model_sources.effective_roots(str(tmp_path / "portable"))

    assert result == (models,)


def test_selected_model_directory_is_unchanged(tmp_path):
    root = tmp_path / "models"
    (root / "ComfyUI" / "models").mkdir(parents=True)

    result = model_sources.effective_roots(str(root))

    assert result == (root,)


def test_generic_packages_directory_does_not_identify_a_library(tmp_path):
    (tmp_path / "Packages").mkdir()
    (tmp_path / "Models").mkdir()

    result = model_sources.effective_roots(str(tmp_path))

    assert result == (tmp_path,)


def test_missing_and_inaccessible_roots_do_not_block_later_root(
    tmp_path, weight, monkeypatch
):
    inaccessible = tmp_path / "denied"
    expected = make_weight(tmp_path / "library" / weight.filename)
    walk = os.walk

    def permission_walk(root, **kwargs):
        if Path(root) == inaccessible:
            return iter(())
        return walk(root, **kwargs)

    monkeypatch.setattr(model_sources.os, "walk", permission_walk)

    result = discover(
        weight,
        [str(tmp_path / "missing"), str(inaccessible), str(expected.parent)],
        tmp_path / "downloads",
    )

    assert result == str(expected)


def test_repo_suffix_beats_filename_only(tmp_path, weight):
    root = tmp_path / "library"
    make_weight(root / "elsewhere" / weight.filename)
    expected = make_weight(root / weight.path_in_repo)

    result = discover(weight, [str(root)], tmp_path / "downloads")

    assert result == str(expected)


def test_ambiguous_candidates_remain_unresolved(tmp_path, weight):
    root = tmp_path / "library"
    for directory in ("a", "b"):
        make_weight(root / directory / weight.filename)

    result = discover(weight, [str(root)], tmp_path / "downloads")

    assert result is model_sources.AMBIGUOUS


def test_ambiguous_higher_priority_root_stops_search(tmp_path, weight, monkeypatch):
    first = tmp_path / "first"
    for directory in ("a", "b"):
        make_weight(first / directory / weight.filename)
    lower = make_weight(tmp_path / "second" / weight.filename)
    cached = make_weight(tmp_path / "cache" / weight.filename)
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: str(cached))

    result = discover(
        weight, [str(first), str(lower.parent)], tmp_path / "downloads"
    )

    assert result is model_sources.AMBIGUOUS


def test_ambiguous_weight_is_not_downloaded(tmp_path, weight):
    root = tmp_path / "library"
    for directory in ("a", "b"):
        make_weight(root / directory / weight.filename)
    prefs = {"model_search_roots": [str(root)]}
    logs = []

    result = fetch_models.fetch_weight(
        weight, str(tmp_path / "downloads"), prefs, log=logs.append
    )

    assert result is False
    assert weight.pref_key not in prefs
    assert any("[ambiguous]" in line for line in logs)


def test_generic_diffusers_filename_from_another_repo_is_rejected(tmp_path):
    weight = fetch_models.Weight(
        "generic", "Qwen/Qwen-Image-2.1", "vae/diffusion_pytorch_model.safetensors",
        1024 / 1024**3, "",
    )
    make_weight(
        tmp_path / "library" / "some-other-diffusers-repo" / "vae" / weight.filename
    )

    result = model_sources.search_model_roots(
        weight, [str(tmp_path / "library")], fetch_models._valid_safetensors, 819
    )

    assert result is None


def test_generic_diffusers_filename_requires_repository_identity(tmp_path):
    weight = fetch_models.Weight(
        "generic", "Qwen/Qwen-Image-2.1", "vae/diffusion_pytorch_model.safetensors",
        1024 / 1024**3, "",
    )
    expected = make_weight(
        tmp_path / "library" / "Qwen" / "Qwen-Image-2.1" / weight.path_in_repo
    )

    result = model_sources.search_model_roots(
        weight, [str(tmp_path / "library")], fetch_models._valid_safetensors, 819
    )

    assert result == str(expected)


def test_wrapper_subtrees_are_compared_together(tmp_path, weight):
    for application in ("ComfyUI", "Fooocus"):
        make_weight(tmp_path / application / "models" / weight.path_in_repo)

    result = discover(weight, [str(tmp_path)], tmp_path / "downloads")

    assert result is model_sources.AMBIGUOUS


def test_original_filename_requires_repo_suffix_when_local_name_differs(
    tmp_path, weight
):
    weight.filename = "specific-model.safetensors"
    make_weight(tmp_path / "library" / "unrelated" / "model.safetensors")

    result = discover(weight, [str(tmp_path / "library")], tmp_path / "downloads")

    assert result is None


def test_original_repo_path_is_usable_with_local_filename_override(tmp_path, weight):
    weight.filename = "specific-model.safetensors"
    expected = make_weight(tmp_path / "library" / weight.path_in_repo)

    result = discover(weight, [str(tmp_path / "library")], tmp_path / "downloads")

    assert result == str(expected)


@pytest.mark.parametrize("invalid", ["header", "small", "large"])
def test_invalid_candidate_is_not_linked(tmp_path, weight, invalid):
    candidate = make_weight(tmp_path / "library" / weight.filename)
    if invalid == "header":
        candidate.write_bytes(b"not safetensors".ljust(1024, b" "))
    else:
        make_weight(candidate, size=100 if invalid == "small" else 2048)

    result = discover(weight, [str(candidate.parent)], tmp_path / "downloads")

    assert result is None


def test_exact_hf_cache_hit_prevents_download(tmp_path, weight, monkeypatch):
    from huggingface_hub import try_to_load_from_cache

    cache = tmp_path / "hub"
    repository = cache / "models--owner--repo"
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text("revision")
    expected = make_weight(repository / "snapshots" / "revision" / weight.path_in_repo)
    monkeypatch.setattr(
        model_sources,
        "try_to_load_from_cache",
        lambda repo, filename: try_to_load_from_cache(repo, filename, cache_dir=cache),
    )
    prefs = {}

    result = fetch_models.fetch_weight(weight, str(tmp_path / "downloads"), prefs)

    assert result is True
    assert prefs[weight.pref_key] == str(expected)
    assert not (tmp_path / "downloads").exists()


@pytest.mark.parametrize("cached", [None, object(), "/missing/cache.safetensors"])
def test_hf_cache_miss_falls_through_to_own_models(
    tmp_path, weight, monkeypatch, cached
):
    expected = make_weight(tmp_path / "downloads" / "nested" / weight.filename)
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: cached)

    result = discover(weight, [], tmp_path / "downloads")

    assert result == str(expected)


def test_known_store_precedes_own_directory(tmp_path, weight, monkeypatch):
    store = tmp_path / "shared" / "Models"
    expected = make_weight(store / weight.filename)
    make_weight(tmp_path / "downloads" / weight.filename)
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: (store,))

    result = discover(weight, [], tmp_path / "downloads")

    assert result == str(expected)


def test_stability_matrix_library_pointer(tmp_path, monkeypatch):
    library = tmp_path / "shared-library"
    models = library / "Models"
    models.mkdir(parents=True)
    pointer = tmp_path / "config" / "StabilityMatrix" / "library.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"LibraryPath": str(library)}))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "config"))
    result = automatic_roots()

    assert result == (models,)


@pytest.mark.parametrize(
    "contents", ["invalid JSON", "[]", '{"LibraryPath": "relative"}']
)
def test_unusable_library_pointer_is_ignored(tmp_path, monkeypatch, contents):
    pointer = tmp_path / "config" / "StabilityMatrix" / "library.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(contents)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "config"))

    result = automatic_roots()

    assert result == ()


def test_unresolved_weight_uses_existing_download_and_persists(
    tmp_path, weight, monkeypatch
):
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(make_weight(Path(kwargs["local_dir"]) / kwargs["filename"]))

    fetch_models._save_prefs(
        str(tmp_path / "prefs.json"),
        {"model_search_roots": [str(tmp_path / "missing")]},
    )
    monkeypatch.setattr(fetch_models, "FAMILIES", {"test": [weight]})
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    result = fetch_models.fetch(["test"], repo_dir=str(tmp_path), log=lambda _: None)

    assert result is True
    assert len(calls) == 1
    prefs = fetch_models._load_prefs(str(tmp_path / "prefs.json"))
    assert prefs[weight.pref_key] == str(tmp_path / "models" / weight.filename)
    assert prefs["model_search_roots"] == [str(tmp_path / "missing")]


def test_unresolved_dry_run_does_not_download_or_write_prefs(
    tmp_path, weight, monkeypatch
):
    monkeypatch.setattr(fetch_models, "FAMILIES", {"test": [weight]})

    result = fetch_models.fetch(
        ["test"], repo_dir=str(tmp_path), log=lambda _: None, dry_run=True
    )

    assert result is True
    assert not (tmp_path / "prefs.json").exists()
    assert not (tmp_path / "models").exists()


def test_tools_only_flow_does_not_discover_path_backed_weights(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        fetch_models, "fetch_tool", lambda spec, **_: calls.append(spec)
    )

    result = fetch_models.fetch(["tools"], repo_dir=str(tmp_path), log=lambda _: None)

    assert result is True
    assert calls == fetch_models.TOOLS
    assert not (tmp_path / "prefs.json").exists()
