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
