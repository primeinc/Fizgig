"""Model validator must distinguish exact known bytes, custom files and wrong components."""
import hashlib
import json
import struct
from types import SimpleNamespace

from fizgig.web.model_validation import validate_path, _fingerprint, _inspect_directory


def _model(role="dit", repo="official/repo", path="weights.safetensors", **extra):
    return SimpleNamespace(role=role, repo=repo, path=path, inside="",
                           alt_repo="", alt_path="", **extra)


def _safetensors(path, keys):
    header = {}
    offset = 0
    values = bytearray()
    for key in keys:
        values.extend(b"\x01\x02\x03\x04")
        header[key] = {"dtype": "U8", "shape": [4], "data_offsets": [offset, offset + 4]}
        offset += 4
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + values)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_known_bytes_pass_without_loading_tensor(tmp_path):
    file = tmp_path / "arbitrary-name.safetensors"
    want = _safetensors(file, ["transformer_blocks.0.attn.to_q.weight"])
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": want, "size": file.stat().st_size, "revision": "a" * 40})
    assert result["state"] == "pass"
    assert result["match"]["repo"] == "official/repo"


def test_wrong_bytes_are_not_passed_even_with_valid_header(tmp_path):
    file = tmp_path / "model.safetensors"
    _safetensors(file, ["transformer_blocks.0.attn.to_q.weight"])
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "0" * 64, "size": file.stat().st_size, "revision": "b" * 40})
    assert result["state"] == "different"


def test_no_reference_is_explicitly_unverified(tmp_path):
    file = tmp_path / "model.safetensors"
    _safetensors(file, ["transformer_blocks.0.attn.to_q.weight"])
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "", "size": file.stat().st_size, "revision": "c" * 40})
    assert result["state"] == "unverified"


def test_lora_cannot_pass_as_dit(tmp_path):
    file = tmp_path / "accidental.safetensors"
    _safetensors(file, ["transformer_blocks.0.attn.to_q.lora_A.weight"])
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "0" * 64, "size": file.stat().st_size})
    assert result["state"] == "invalid"
    assert "Wrong component" in result["message"]


def test_wrong_size_short_circuits_full_hash(tmp_path):
    file = tmp_path / "model.safetensors"
    _safetensors(file, ["transformer_blocks.0.attn.to_q.weight"])
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "0" * 64, "size": 999999999})
    assert result["state"] == "different"


def test_invalid_header_is_rejected(tmp_path):
    file = tmp_path / "broken.safetensors"
    file.write_bytes(b"not a safetensors file")
    result = validate_path(str(file), _model(), lookup=lambda *_: {})
    assert result["state"] == "invalid"


def test_missing_shard_fails(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"example": "missing-00001.safetensors"}}))
    assert _inspect_directory(str(root))["state"] == "invalid"


def test_unknown_tensor_kind_is_not_guessed():
    assert _fingerprint(["unfamiliar.model.weight"]) == "unknown"



def _keys(n=48):
    return [f"transformer_blocks.{i}.attn.to_q.weight" for i in range(n)]


def test_custom_weights_with_matching_architecture_are_not_called_wrong_model(tmp_path):
    file = tmp_path / "custom-finetune.safetensors"
    keys = _keys()
    _safetensors(file, keys)
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "0" * 64, "size": file.stat().st_size,
        "revision": "d" * 40, "tensors": {key: [4] for key in keys}})
    assert result["state"] == "layout_match"
    assert result["structure"]["matched"] == len(keys)
    assert "NOT verified" in result["message"]


def test_different_model_architecture_is_detected_by_tensor_names(tmp_path):
    file = tmp_path / "actually-other-family.safetensors"
    keys = _keys()
    _safetensors(file, keys)
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "0" * 64, "size": file.stat().st_size,
        "tensors": {f"completely_other_family.{i}.weight": [4] for i in range(48)}})
    assert result["state"] == "layout_mismatch"
    assert result["structures"][0]["common"] == 0


def test_similar_tensor_names_with_wrong_dimensions_are_rejected(tmp_path):
    file = tmp_path / "wrong-size-architecture.safetensors"
    keys = _keys()
    _safetensors(file, keys)
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "0" * 64, "size": file.stat().st_size,
        "tensors": {key: [999] for key in keys}})
    assert result["state"] == "layout_mismatch"
    assert result["structures"][0]["wrong_shapes"] == len(keys)


def test_tiny_matching_header_is_not_claimed_compatible(tmp_path):
    file = tmp_path / "tiny.safetensors"
    _safetensors(file, ["transformer_blocks.0.weight"])
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "0" * 64, "size": file.stat().st_size,
        "tensors": {"transformer_blocks.0.weight": [4]}})
    assert result["state"] == "different"  # insufficient structural evidence


def test_layout_match_tolerates_weight_dtype_changes_but_not_shape_changes(tmp_path):
    file = tmp_path / "quantized.safetensors"
    keys = _keys()
    _safetensors(file, keys)
    result = validate_path(str(file), _model(), lookup=lambda *_: {
        "sha256": "0" * 64, "size": 1000000,
        "tensors": {key: [4] for key in keys}})
    assert result["state"] == "layout_match"



def test_dit_file_is_rejected_in_training_adapter_slot(tmp_path):
    file = tmp_path / "not-an-adapter.safetensors"
    _safetensors(file, _keys(24))
    result = validate_path(str(file), _model(role="training_adapter"),
                           lookup=lambda *_: {"sha256": "0" * 64})
    assert result["state"] == "invalid"
    assert "Wrong component" in result["message"]


def test_pinned_reference_conflict_cannot_be_approved_by_new_hub_commit(tmp_path):
    file = tmp_path / "text-encoder.safetensors"
    keys = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(24)]
    _safetensors(file, keys)
    build = SimpleNamespace(repo="community/encoder", path="text.safetensors",
                            label="verified build", sha256="1" * 64)
    result = validate_path(str(file), _model(role="text_encoder", repo="", path=""),
                           builds=[build], lookup=lambda *_: {
                               "sha256": "2" * 64, "size": file.stat().st_size})
    assert result["state"] == "unverified"
    assert "conflicts" in result["reference_errors"][0]


def test_declared_checksum_survives_missing_hub_digest(tmp_path):
    file = tmp_path / "text-encoder.safetensors"
    keys = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(24)]
    digest = _safetensors(file, keys)
    build = SimpleNamespace(repo="community/encoder", path="text.safetensors",
                            label="known build", sha256=digest)
    result = validate_path(str(file), _model(role="text_encoder", repo="", path=""),
                           builds=[build], lookup=lambda *_: {"sha256": "", "size": file.stat().st_size})
    assert result["state"] == "pass"
