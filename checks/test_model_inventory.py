"""Status/size provenance for the compact Preferences model-path workspace."""
from pathlib import Path
from types import SimpleNamespace

from fizgig.web.model_inventory import GIB, inventory


def file(key, size_gb=1.0, required=True, same_as=""):
    return SimpleNamespace(pref_key=key, size_gb=size_gb, required=required, same_as=same_as)


def family(*models):
    return SimpleNamespace(model_files=models)


def test_sizes_are_actual_bytes_and_missing_weights_are_estimates(tmp_path):
    existing = tmp_path / "models" / "qwen_image_2.1_bf16.safetensors"
    existing.parent.mkdir()
    existing.write_bytes(b"already downloaded model")
    prefs = {"dit": str(existing), "vae": "", "optional": ""}
    result = inventory(prefs, [family(file("dit", 13.3), file("vae", 1.35),
                                      file("optional", 1.2, required=False))], str(tmp_path))
    files, summary = result["files"], result["summary"]
    assert files["dit"]["exists"]
    assert files["dit"]["bytes"] == existing.stat().st_size
    assert files["dit"]["expected_bytes"] > files["dit"]["bytes"]  # NOT a partial download!
    assert summary["disk_bytes"] == existing.stat().st_size
    assert summary["remaining_required_bytes"] == round(1.35 * GIB)
    assert summary["remaining_optional_bytes"] == round(1.2 * GIB)
    assert (summary["found"], summary["missing"], summary["stale"], summary["optional"]) == (1, 2, 0, 1)


def test_shared_path_and_shared_pref_are_counted_once(tmp_path):
    existing = tmp_path / "models" / "shared.safetensors"
    existing.parent.mkdir()
    existing.write_bytes(b"shared across families")
    result = inventory({"one": str(existing), "two": str(existing)},
                       [family(file("one"), file("two")), family(file("one"))], str(tmp_path))
    assert result["summary"]["count"] == 2
    assert result["summary"]["found"] == 2
    assert result["summary"]["disk_bytes"] == existing.stat().st_size


def test_stale_path_is_not_reported_on_disk(tmp_path):
    stale = tmp_path / "models" / "not-there.safetensors"
    result = inventory({"dit": str(stale)}, [family(file("dit", 13.3))], str(tmp_path))
    assert result["files"]["dit"]["configured"] and not result["files"]["dit"]["exists"]
    assert result["summary"]["stale"] == 1
    assert result["summary"]["disk_bytes"] == 0
    assert result["summary"]["remaining_required_bytes"] == round(13.3 * GIB)


def test_relative_path_resolves_against_repo(tmp_path):
    f = tmp_path / "user_data" / "models" / "model.safetensors"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"valid path")
    result = inventory({"dit": "user_data/models/model.safetensors"},
                       [family(file("dit"))], str(tmp_path))
    assert result["files"]["dit"]["exists"]
    assert result["files"]["dit"]["bytes"] == len(b"valid path")


def test_configured_optional_not_included_in_required_remaining(tmp_path):
    data = inventory({"speed": ""}, [family(file("speed", 1.19, False))], str(tmp_path))
    assert data["summary"]["remaining_required_bytes"] == 0
    assert data["summary"]["remaining_optional_bytes"] == round(1.19 * GIB)


def test_paths_from_family_options_are_included_even_without_model_file_metadata(tmp_path):
    model = tmp_path / "custom" / "extra.safetensors"
    model.parent.mkdir()
    model.write_bytes(b"known existing file")
    data = inventory(
        {"dit": "", "option_model": str(model)},
        [family(file("dit", 13.3))], str(tmp_path),
        key_resolver=lambda _: ["dit", "option_model"])
    assert data["files"]["option_model"]["exists"]
    assert data["files"]["option_model"]["bytes"] == len(b"known existing file")
    assert data["summary"]["disk_bytes"] == len(b"known existing file")


def test_directory_models_count_ready_without_expensive_recursive_walk(tmp_path):
    directory = tmp_path / "krea2" / "text_encoder"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"a model shard")
    result = inventory({"encoder": str(directory)}, [family(file("encoder", 8.88))], str(tmp_path))
    assert result["files"]["encoder"]["exists"]
    assert result["files"]["encoder"]["directory"]
    assert result["summary"]["found"] == 1
    assert result["summary"]["folder_count"] == 1
    assert result["summary"]["disk_bytes"] == 0  # directory content not falsely reported as 0 GB total
    assert result["summary"]["remaining_required_bytes"] == 0


def test_shared_missing_model_counts_only_one_download(tmp_path):
    # Z-Image Turbo/Base reuse the exact same VAE and Qwen3 text encoder.
    f1 = family(file("zimage_vae", 0.34), file("zimage_text_encoder", 8.04))
    f2 = family(file("zimage_base_vae", 0.34, same_as="zimage_vae"),
                file("zimage_base_text_encoder", 8.04, same_as="zimage_text_encoder"))
    data = inventory({}, [f1, f2], str(tmp_path))
    assert data["summary"]["count"] == 4
    assert data["summary"]["remaining_required_bytes"] == round(0.34 * GIB) + round(8.04 * GIB)


def test_shared_model_available_under_other_family_needs_no_download(tmp_path):
    path = tmp_path / "zimage_vae.safetensors"
    path.write_bytes(b"shared model")
    a = file("vae", 0.34)
    b = file("base_vae", 0.34, same_as="vae")
    data = inventory({"vae": str(path)}, [family(a), family(b)], str(tmp_path))
    assert data["summary"]["remaining_required_bytes"] == 0
    assert data["summary"]["disk_bytes"] == path.stat().st_size
