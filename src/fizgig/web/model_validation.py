"""Provenance-aware model-file checks for Preferences.

A safetensors header is a *format* check, not a model identity check. A PASS
requires a whole-file SHA-256 match against the family catalog's listed build,
or a catalog-approved text encoder alternative. No GPU/model weights are loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

_LOCK = threading.RLock()
_JOBS = {}
_ACTIVE = None
_STOP = None
_CHUNK = 8 * 1024 * 1024


class Cancelled(Exception):
    """The user cancelled an in-progress disk scan."""


def _identity(path):
    st = os.stat(path)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _fingerprint(keys):
    """Only assert obvious component kinds. Ambiguous architectures stay unknown."""
    names = tuple(keys)
    if any(".lora_A." in k or ".lora_B." in k or ".lora_down." in k
           or ".lora_up." in k or "lokr_w1" in k or "hada_w1" in k
           or k.startswith(("lora_unet_", "lora_te")) for k in names):
        return "lora"
    diffusion = any(k.startswith(("model.diffusion_model.", "diffusion_model."))
                    for k in names)
    vae = any(k.startswith(("first_stage_model.encoder.", "encoder.")) for k in names) and any(
        k.startswith(("first_stage_model.decoder.", "decoder.")) for k in names)
    if diffusion and vae:
        return "checkpoint"
    if vae and not diffusion:
        return "vae"
    if any(k.startswith(("double_blocks.", "single_blocks.", "transformer_blocks.",
                         "joint_blocks.", "model.diffusion_model.", "diffusion_model."))
           for k in names):
        return "dit"
    if any(k.startswith(("model.layers.", "language_model.model.layers.",
                         "text_model.encoder.layers.", "transformer.h."))
           for k in names):
        return "text_encoder"
    return "unknown"


def _wrong_kind(kind, role, inside=""):
    if kind == "unknown" or not role:
        return False
    expected = {
        "dit": {"dit", "checkpoint"}, "ref_dit": {"dit", "checkpoint"},
        "preview_dit": {"dit", "checkpoint"},
        "vae": {"vae"}, "audio_vae": {"vae"},
        "text_encoder": {"text_encoder"},
        "speed_lora": {"lora"}, "look_lora": {"lora"},
        "training_adapter": {"lora"}, "model_eq": {"lora"},
    }.get(role)
    # A single-file checkpoint legitimately supplies SDXL's VAE/encoders.
    if inside and kind == "checkpoint":
        return False
    # Distinct model families use nonstandard tensor names, so only reject
    # unmistakable LoRA-for-base / base-for-LoRA or VAE-for-base swaps.
    if kind == "lora" and role not in ("speed_lora", "look_lora", "training_adapter", "model_eq"):
        return True
    if kind in ("vae", "dit", "checkpoint") and role in ("speed_lora", "look_lora", "model_eq"):
        return True
    if kind == "vae" and role in ("dit", "ref_dit", "preview_dit", "text_encoder"):
        return True
    if kind == "text_encoder" and role in ("vae", "audio_vae", "dit", "ref_dit", "preview_dit"):
        return True
    return False


def _inspect_file(path):
    if not path.lower().endswith(".safetensors"):
        return {"state": "unverified", "message": "This file is not safetensors. It was not deserialized or executed.",
                "kind": "unknown", "tensors": 0}
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt", device="cpu") as opened:
            keys = list(opened.keys())
            if not keys:
                return {"state": "invalid", "message": "Safetensors file contains no tensors."}
            key = keys[0]
            part = opened.get_slice(key)
            example = {"name": key, "shape": list(part.get_shape()), "dtype": part.get_dtype()}
            kind = _fingerprint(keys)
        return {"state": "inspected", "kind": kind, "tensors": len(keys), "example": example}
    except (OSError, ValueError, RuntimeError, Exception) as exc:
        return {"state": "invalid", "message": "Cannot read safetensors header: " + str(exc)[:250]}


def _inspect_directory(path):
    """Reject empty/incomplete HF shard folders without importing a model."""
    indexes = ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
    shards = []
    for name in indexes:
        index_path = os.path.join(path, name)
        if not os.path.isfile(index_path):
            continue
        if os.path.getsize(index_path) > 8 * 1024 * 1024:
            return {"state": "invalid", "message": "Shard index is too large to inspect safely."}
        try:
            with open(index_path, encoding="utf-8") as f:
                manifest = json.load(f)
            mapping = manifest["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("empty weight_map")
            shards = sorted(set(mapping.values()))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return {"state": "invalid", "message": "Invalid shard index: " + str(exc)[:200]}
        break
    if not shards:
        try:
            shards = sorted(f for f in os.listdir(path) if f.lower().endswith(".safetensors"))
        except OSError as exc:
            return {"state": "invalid", "message": str(exc)[:200]}
    if not shards:
        return {"state": "invalid", "message": "No safetensors weights or shard index in this folder."}
    if len(shards) > 256:
        return {"state": "unverified", "message": "More than 256 shards; no complete folder verification was run.",
                "shards": len(shards)}
    root = os.path.realpath(path)
    for name in shards:
        if not isinstance(name, str) or not name.lower().endswith(".safetensors"):
            return {"state": "invalid", "message": "Shard index references a non-safetensors file."}
        full = os.path.realpath(os.path.join(root, name))
        if os.path.commonpath((root, full)) != root or not os.path.isfile(full):
            return {"state": "invalid", "message": "Missing or unsafe shard: " + name[:160]}
        inspected = _inspect_file(full)
        if inspected["state"] == "invalid":
            return {"state": "invalid", "message": name[:100] + ": " + inspected["message"]}
    return {"state": "unverified", "message": f"{len(shards)} shard(s) parse, but the folder's model identity and loader compatibility are unverified.",
            "shards": len(shards)}


def _sources(model, builds=(), alias_models=()):
    result = []
    known = set()

    def add(repo, path, label, sha256=""):
        if not repo or not path or (repo, path) in known:
            return
        known.add((repo, path))
        result.append({"repo": repo, "path": path, "label": label, "sha256": sha256})

    if model is not None:
        add(getattr(model, "repo", ""), getattr(model, "path", ""), "Catalog build")
        add(getattr(model, "alt_repo", ""), getattr(model, "alt_path", ""), "Alternative build")
        for sibling in alias_models:
            add(getattr(sibling, "repo", ""), getattr(sibling, "path", ""), "Shared-family build")
            add(getattr(sibling, "alt_repo", ""), getattr(sibling, "alt_path", ""),
                "Shared-family alternative")
        if getattr(model, "role", "") == "text_encoder":
            for build in builds:
                add(build.repo, build.path, build.label, getattr(build, "sha256", ""))
    return result


def _hub_reference(repo, path):
    """Resolve a *specific* Hub commit before reading one file's LFS metadata."""
    from huggingface_hub import HfApi
    api = HfApi()
    model = api.model_info(repo, timeout=12)
    if not model.sha:
        raise ValueError("Hub did not return a model revision")
    files = api.get_paths_info(repo, [path], revision=model.sha)
    info = next((item for item in files if getattr(item, "path", None) == path), None)
    if info is None:
        raise ValueError("The catalog path is absent from the official repository")
    lfs = getattr(info, "lfs", None)
    return {"sha256": getattr(lfs, "sha256", "") if lfs else "",
            "size": getattr(info, "size", None), "revision": model.sha}


def _hash(path, size, progress, cancel):
    digest = hashlib.sha256()
    count = 0
    with open(path, "rb") as source:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            chunk = source.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            count += len(chunk)
            if progress and (count % (8 * _CHUNK) == 0 or count >= size):
                progress("hashing", count, size)
    if count != size:
        raise ValueError("The file changed size during verification.")
    return digest.hexdigest()


def validate_path(path, model, *, builds=(), alias_models=(), lookup=None,
                  progress=None, cancel=None):
    """Bounded file inspection + full SHA-256 against declared/Hub-proven builds.

    PASS means exact official/catalog-approved file bytes, not that a GPU load ran.
    Different and unverified are distinct outcomes; neither is marked PASS.
    """
    if not isinstance(path, str) or not path.strip() or len(path) > 4096 or "\0" in path:
        return {"state": "invalid", "message": "Select one model path to test."}
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.exists(path):
        return {"state": "invalid", "message": "File or folder does not exist.", "path": path}
    if os.path.isdir(path):
        return {**_inspect_directory(path), "path": path}
    if not os.path.isfile(path):
        return {"state": "invalid", "message": "Not a regular model file.", "path": path}
    before = _identity(path)
    structural = _inspect_file(path)
    if structural["state"] != "inspected":
        return {**structural, "path": path}
    role = getattr(model, "role", "") if model is not None else ""
    if _wrong_kind(structural["kind"], role, getattr(model, "inside", "") if model else ""):
        return {"state": "invalid", "path": path, **{k: v for k, v in structural.items() if k != "state"},
                "message": f"Wrong component type: this looks like {structural['kind']}, not {role}."}
    if cancel is not None and cancel.is_set():
        return {"state": "cancelled", "path": path, "message": "Test cancelled."}
    refs = _sources(model, builds, alias_models)
    if not refs:
        return {"state": "unverified", "path": path, **structural,
                "message": "Valid safetensors layout, but this slot has no trusted reference checksum."}
    resolve = lookup or _hub_reference
    errors = []
    candidates = []
    if progress:
        progress("reference", 0, before[2])
    for item in refs:
        if cancel is not None and cancel.is_set():
            return {"state": "cancelled", "path": path, "message": "Test cancelled."}
        source = dict(item)
        try:
            remote = resolve(item["repo"], item["path"])
            source.update({k: v for k, v in remote.items() if v is not None})
        except Exception as exc:
            errors.append(item["label"] + ": " + str(exc)[:110])
        if isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64:
            candidates.append(source)
    if not candidates:
        return {"state": "unverified", "path": path, **{k: v for k, v in structural.items() if k != "state"},
                "message": "Safetensors header parses, but no trusted SHA-256 was available; identity NOT verified.",
                "reference_errors": errors[:3]}
    plausible = [c for c in candidates if not isinstance(c.get("size"), int) or c["size"] == before[2]]
    if not plausible:
        return {"state": "different", "path": path, **{k: v for k, v in structural.items() if k != "state"},
                "message": "Wrong file size for every catalog-approved build. This is not a verified model for this slot.",
                "actual_bytes": before[2],
                "expected": [{"label": c["label"], "bytes": c["size"]} for c in candidates]}
    if progress:
        progress("hashing", 0, before[2])
    try:
        digest = _hash(path, before[2], progress, cancel)
    except Cancelled:
        return {"state": "cancelled", "path": path, "message": "Test cancelled."}
    except (OSError, ValueError) as exc:
        return {"state": "invalid", "path": path, "message": str(exc)}
    if _identity(path) != before:
        return {"state": "invalid", "path": path, "message": "The file was modified while it was being tested."}
    for candidate in plausible:
        if digest.lower() == candidate["sha256"].lower():
            return {"state": "pass", "path": path, **{k: v for k, v in structural.items() if k != "state"},
                    "message": "Exact SHA-256 match. File identity VERIFIED; runtime GPU loading not tested.",
                    "match": {k: candidate.get(k) for k in ("repo", "path", "label", "revision")},
                    "sha256": digest, "bytes": before[2]}
    return {"state": "different", "path": path, **{k: v for k, v in structural.items() if k != "state"},
            "message": "SHA-256 differs from every catalog-approved build. This may be a custom model, but it is NOT verified for this slot.",
            "sha256": digest, "checked": [c["label"] for c in plausible]}


def _catalog(family_key, key):
    from fizgig.families import form, registry
    desc = registry.get(family_key)
    if not desc or key not in form.model_keys(desc):
        raise ValueError("Unknown model family or model slot.")
    model = next((m for m in desc.model_files if m.pref_key == key), None)
    shared = []
    if model is not None and getattr(model, "same_as", ""):
        for family in registry.training_families():
            shared.extend(m for m in family.model_files if m.pref_key == model.same_as)
    return model, getattr(desc, "text_encoder_builds", ()), shared


def status(family, key):
    with _LOCK:
        return dict(_JOBS.get((family, key), {"state": "idle"}))


def start(family, key, path, *, lookup=None):
    global _ACTIVE, _STOP
    model, builds, shared = _catalog(family, key)
    if not isinstance(path, str) or len(path) > 4096:
        raise ValueError("Choose a valid model path.")
    with _LOCK:
        if _ACTIVE is not None:
            raise ValueError("Another model test is running. Wait or cancel it first.")
        slot = (family, key)
        _ACTIVE = slot
        stop = threading.Event()
        _STOP = stop
        _JOBS[slot] = {"state": "running", "stage": "inspect", "path": path, "done": 0, "total": 0}

        def progress(stage, done, total):
            with _LOCK:
                if _ACTIVE == slot:
                    _JOBS[slot].update(stage=stage, done=done, total=total)

        def work():
            global _ACTIVE, _STOP
            try:
                outcome = validate_path(path, model, builds=builds, alias_models=shared,
                                        lookup=lookup, progress=progress, cancel=stop)
            except Exception as exc:
                outcome = {"state": "error", "message": "Model test failed: " + str(exc)[:250]}
            with _LOCK:
                _JOBS[slot] = outcome
                if _ACTIVE == slot:
                    _ACTIVE, _STOP = None, None

        threading.Thread(target=work, daemon=True, name="fizgig-model-verify").start()
        return dict(_JOBS[slot])


def cancel(family, key):
    with _LOCK:
        if _ACTIVE != (family, key) or _STOP is None:
            return False
        _STOP.set()
        return True
