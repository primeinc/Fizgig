"""Search Locally also finds the one-off MOSS preview LoRAs not in bulk downloads.

Regression for the case where both files are present under the user's
user_data/models/Lora folder but Search Locally leaves their prefs empty.
"""
from fizgig import model_sources
from fizgig.families.registry import FAMILIES
from fizgig.scripts import fetch_models


def test_optional_moss_loras_are_discoverable_not_bulk_downloadable():
    family = FAMILIES["moss_image"]
    optional = {f.pref_key for f in family.model_files if not f.bulk_fetch and f.repo and f.path}
    assert {"moss_image_turbo_lora", "moss_image_sft_look_lora"} <= optional
    findable = {w.pref_key for w in fetch_models.discovery_candidates(("moss_image",))}
    bulk = {w.pref_key for w in fetch_models.model_candidates(("moss_image",))}
    assert optional <= findable
    assert optional.isdisjoint(bulk)


def test_moss_lora_files_in_uppercase_lora_folder_are_reused(tmp_path, monkeypatch):
    own = tmp_path / "Fizgig" / "user_data" / "models"
    lora = own / "Lora"
    lora.mkdir(parents=True)
    names = {}
    candidates = {w.pref_key: w for w in fetch_models.discovery_candidates(("moss_image",))}
    for key in ("moss_image_turbo_lora", "moss_image_sft_look_lora"):
        w = candidates[key]
        path = lora / w.filename
        path.write_bytes(b"synthetic model")
        names[key] = path

    monkeypatch.setattr(fetch_models, "_valid_safetensors", lambda path, _: True)
    monkeypatch.setattr(model_sources, "automatic_roots", lambda: ())
    monkeypatch.setattr(model_sources, "try_to_load_from_cache", lambda *_: None)
    for key, path in names.items():
        found = fetch_models.find_weight(candidates[key], str(own), {"model_search_roots": [str(own)]})
        assert found == str(path)


def test_manual_alternative_stays_a_link_not_an_implicit_download_choice():
    # The Qwen3.5 int8 link is a manual alternative, not a fetch_choice.
    # Discovery must not silently swap it in when given an unsupported choice.
    desc = FAMILIES["moss_image"]
    encoder = next(m for m in desc.model_files if m.pref_key == "moss_image_text_encoder")
    assert encoder.alt_path.endswith("qwen3.5_9b_int8_convrot.safetensors")
    assert not encoder.fetch_choice
    candidates = {w.pref_key: w for w in fetch_models.discovery_candidates(
        ("moss_image",), choices={"moss_image_text_encoder": "alt"})}
    assert candidates["moss_image_text_encoder"].filename == "qwen3.5_9b_bf16.safetensors"
