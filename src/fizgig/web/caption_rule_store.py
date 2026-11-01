"""Per-training-folder caption replacement rules.

Kept in Fizgig user_data, not inside image libraries. Shared by Captions ->
Caption All and the training loss-watch. Reuses caption_replace's literal parser.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

from fizgig.caption_replace import normalized_saved_rules
from fizgig.user_data import user_file

_LOCK = threading.RLock()


def _location(folder: str) -> str:
    if not isinstance(folder, str) or not folder.strip():
        raise ValueError("Choose a training folder on the Start tab first.")
    full = os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(folder.strip()))))
    if not os.path.isdir(full):
        raise ValueError("The selected training folder does not exist.")
    digest = hashlib.sha256(os.fsencode(full)).hexdigest()
    return os.path.join(user_file("caption_auto_rules"), digest + ".json")


def load(folder: str) -> dict:
    """Read a folder's own rules. Never borrow them from another dataset."""
    if not folder:
        return {"enabled": False, "rules": []}
    path = _location(folder)
    with _LOCK:
        try:
            with open(path, encoding="utf-8") as stream:
                source = json.load(stream)
        except FileNotFoundError:
            return {"enabled": False, "rules": []}
    if not isinstance(source, dict):
        raise ValueError("The saved caption rules have an invalid format.")
    return {"enabled": source.get("enabled") is True,
            "rules": normalized_saved_rules(source.get("rules", []))}


def save(folder: str, enabled: bool, rules: list) -> dict:
    """Atomically persist an ordered, bounded rule list for exactly one folder."""
    if not isinstance(enabled, bool):
        raise ValueError("Automatic caption replacement must be on or off.")
    normalized = normalized_saved_rules(rules)
    result = {"enabled": enabled, "rules": normalized}
    path = _location(folder)
    with _LOCK:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        os.replace(temp, path)
    return result
