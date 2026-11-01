"""Where newly fetched path-backed weights are stored.

Existing model paths are never moved. The selected target is an existing local model
library, not a second catalog or an attempt to edit ComfyUI/SwarmUI configuration.

Proven conventional subfolders:
 Comfy-Org/ComfyUI :: folder_paths.py (models/diffusion_models, text_encoders, vae, loras)
 mcmonkeyprojects/SwarmUI :: src/Core/Settings.cs, docs/Model Support.md
 (Models/diffusion_models, text_encoders, VAE, Lora; whole checkpoints use Stable-Diffusion).
"""
from __future__ import annotations

import os
from pathlib import Path


LAYOUTS = ("fizgig", "comfyui", "swarmui")

_SUBDIRS = {
    "comfyui": {
        "diffusion_models": "diffusion_models",
        "checkpoints": "checkpoints",
        "text_encoders": "text_encoders",
        "vae": "vae",
        "loras": "loras",
        "audio_encoders": "audio_encoders",
        "adapters": "fizgig_adapters",
    },
    "swarmui": {
        "diffusion_models": "diffusion_models",
        "checkpoints": "Stable-Diffusion",
        "text_encoders": "text_encoders",
        "vae": "VAE",
        "loras": "Lora",
        "audio_encoders": "audio_encoders",
        "adapters": "fizgig_adapters",
    },
}


def library_root(layout: str, selected: str) -> Path:
    """Accept an application folder or its actual configured model library root.

    A custom SwarmUI ModelRoot is selected directly. Never inspect or rewrite another
    app's settings, and never silently create an absent application's Models folder.
    """
    if layout not in ("comfyui", "swarmui"):
        raise ValueError("Select ComfyUI or SwarmUI to use a shared model library.")
    if not str(selected or "").strip():
        raise ValueError("Choose the model library folder before downloading.")
    root = Path(str(selected).strip()).expanduser().absolute()
    if not root.is_dir():
        raise ValueError(f"Model library folder does not exist: {root}")
    child = "models" if layout == "comfyui" else "Models"
    if root.name.lower() == child.lower():
        return root
    nested = root / child
    if nested.is_dir():
        return nested
    if root.name.lower() in ("comfyui", "swarmui"):
        raise ValueError(f"{root} has no {child} subfolder. Choose its actual model library.")
    # Not a recognized app installation: this is a deliberately chosen custom ModelRoot.
    return root


def model_kind(weight) -> str:
    """Classify from the canonical repo-relative path, then its existing pref key."""
    parts = str(weight.path_in_repo).lower().replace("\\", "/").split("/")
    key = str(weight.pref_key).lower()
    if "training_adapter" in key or "circlestone_adapter" in key:
        return "adapters"
    if any(part in ("loras", "lora") for part in parts) or key.endswith(("_lora", "_look_lora", "_turbo_lora")):
        return "loras"
    if "audio_encoder" in key or "audio_encoders" in parts:
        return "audio_encoders"
    if "vae" in parts or key == "vae" or key.endswith("_vae"):
        return "vae"
    if "text_encoders" in parts or "clip" in parts or "text_encoder" in key:
        return "text_encoders"
    if "checkpoints" in parts:
        return "checkpoints"
    return "diffusion_models"


def destination(weight, default_models_dir, prefs) -> tuple[str, str]:
    """Return (download parent, final absolute file path) without creating anything.

    Generic Diffusers filenames require repo identity to avoid collisions and to
    retain their resolver's strict repo+path match. Ordinary filenames stay flat.
    """
    layout = str(prefs.get("model_download_layout") or "fizgig").strip().lower()
    if layout not in LAYOUTS:
        raise ValueError(f"Unrecognized model download layout: {layout}")
    filename = str(weight.filename)
    if not filename or os.path.basename(filename) != filename or filename in (".", ".."):
        raise ValueError("Model metadata has an unsafe filename")
    if layout == "fizgig":
        base = Path(default_models_dir).absolute()
    else:
        root = library_root(layout, prefs.get("model_download_root", ""))
        base = root / _SUBDIRS[layout][model_kind(weight)]
    if layout != "fizgig" and filename == "diffusion_pytorch_model.safetensors":
        suffix = [*str(weight.repo).split("/"), *str(weight.path_in_repo).split("/")]
        if not suffix or any(not p or p in (".", "..") or "/" in p or "\\" in p for p in suffix):
            raise ValueError("Model metadata has an unsafe repository path")
        final = base.joinpath(*suffix)
    else:
        final = base / filename
    return str(base), str(final.absolute())
