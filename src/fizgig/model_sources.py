"""Cheap, on-demand sources for path-backed weights; no model manifest or index."""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

# (name, subtree, identifying files/directories, obvious wrapper children).
# Defaults: ComfyUI/folder_paths.py, A1111/modules/paths_internal.py,
# SwarmUI/src/Core/Settings.cs, Fooocus/modules/config.py, and
# StabilityMatrix.Core/Services/SettingsManager.cs.
TREE_STORES = (
    ("ComfyUI", "models", (("folder_paths.py",),), ("ComfyUI",)),
    (
        "A1111 / Forge",
        "models",
        (("modules/paths_internal.py",),),
        ("stable-diffusion-webui", "stable-diffusion-webui-forge"),
    ),
    ("SwarmUI", "Models", (("src/Core/Settings.cs",),), ("SwarmUI",)),
    ("Fooocus", "models", (("entry_with_update.py",),), ("Fooocus",)),
    (
        "Stability Matrix",
        "Models",
        ((".sm-portable",), ("settings.json", "Packages")),
        ("StabilityMatrix",),
    ),
)


def effective_roots(selected):
    """Narrow recognized roots without changing the user's persisted selection."""
    root = Path(selected).expanduser().absolute()
    if root.name.casefold() == "models":
        return (root,)
    for _name, subtree, markers, children in TREE_STORES:
        if (
            root.name.casefold() in {name.casefold() for name in children}
            or any(
                all(os.path.exists(root / marker) for marker in group)
                for group in markers
            )
        ) and os.path.isdir(root / subtree):
            return (root / subtree,)
    nested = tuple(
        root / child / subtree
        for _name, subtree, _markers, children in TREE_STORES
        for child in children
        if os.path.isdir(root / child / subtree)
    )
    return nested or (root,)


def automatic_roots():
    """Read Stability Matrix's documented library pointer, not its configuration."""
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        appdata = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    pointer = appdata / "StabilityMatrix" / "library.json"
    try:
        with pointer.open(encoding="utf-8") as stream:
            library = json.loads(stream.read(64 * 1024)).get("LibraryPath")
        if isinstance(library, str) and Path(library).is_absolute():
            models = Path(library) / "Models"
            if models.is_dir():
                return (models,)
    except (OSError, ValueError, AttributeError):
        return ()
    return ()


def _find_in_roots(weight, roots, validate, min_bytes):
    """Prefer a repo-relative suffix, then the local filename; ties are unsafe."""
    suffix = tuple(weight.path_in_repo.split("/"))
    names = {weight.filename, suffix[-1]}
    matches = {}
    entries = (entry for root in roots for entry in os.walk(root, followlinks=False))
    for directory, _dirs, files in entries:
        for name in sorted(names.intersection(files)):
            candidate = Path(directory) / name
            has_suffix = candidate.parts[-len(suffix) :] == suffix
            if name != weight.filename and not has_suffix:
                continue
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            if size > weight.gb * 1.2 * 1024**3 or not validate(candidate, min_bytes):
                continue
            rank = (
                has_suffix,
                name == weight.filename,
            )
            # Alias paths to the same file are not competing identities.
            matches.setdefault(rank, {})[os.path.realpath(candidate)] = candidate
    if not matches:
        return None
    best = matches[max(matches)]
    if len(best) != 1:
        return None
    return str(next(iter(best.values())).absolute())


def resolve_weight(weight, roots, models_dir, validate, min_bytes):
    """User trees, exact HF cache, known stores, then Fizgig's download directory."""
    if isinstance(roots, (list, tuple)):
        for selected in roots:
            if not isinstance(selected, str) or not selected.strip():
                continue
            found = _find_in_roots(
                weight, effective_roots(selected), validate, min_bytes
            )
            if found:
                return found
    try:
        cached = try_to_load_from_cache(weight.repo, weight.path_in_repo)
    except (OSError, ValueError):
        cached = None
    if isinstance(cached, str) and validate(cached, min_bytes):
        return os.path.abspath(cached)
    for root in (*automatic_roots(), Path(models_dir).absolute()):
        found = _find_in_roots(weight, (root,), validate, min_bytes)
        if found:
            return found
    return None
