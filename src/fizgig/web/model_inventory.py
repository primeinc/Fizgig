"""Model-path inventory for Preferences, from canonical FamilyDescription metadata.

File sizes are measured logical bytes. Existing model directories are usable but
not recursively measured: scanning tens of thousands of model files on every
Preferences refresh would be expensive. Missing sizes are approximate.
"""
import os


GIB = 1024 ** 3


def inventory(prefs, families, repo_dir, key_resolver=None):
    """Track every declared model path and deduplicate shared model estimates."""
    entries = {}
    aliases = {}
    for family in families:
        known = {m.pref_key: m for m in family.model_files}
        keys = key_resolver(family) if key_resolver else known
        for key in keys:
            model = known.get(key)
            original = str(prefs.get(key) or "").strip()
            path = os.path.abspath(original if os.path.isabs(original)
                                   else os.path.join(repo_dir, original)) if original else ""
            exists = False
            directory = False
            size = 0
            try:
                if path:
                    directory = os.path.isdir(path)
                    exists = directory or os.path.isfile(path)
                    if exists and not directory:
                        size = os.path.getsize(path)
            except OSError:
                exists, directory, size = False, False, 0
            nominal = max(0, round(float(model.size_gb or 0) * GIB)) if model else 0
            if model and getattr(model, "same_as", ""):
                aliases[key] = model.same_as
            previous = entries.get(key)
            if previous is None:
                entries[key] = {"exists": exists, "directory": directory,
                                "configured": bool(original), "bytes": size,
                                "expected_bytes": nominal,
                                "required": bool(model.required) if model else False,
                                "path": path if exists else ""}
            else:
                previous["required"] |= bool(model.required) if model else False
                previous["expected_bytes"] = max(previous["expected_bytes"], nominal)

    def shared_identity(key):
        """Resolve same_as links without assuming one direction or allowing cycles."""
        seen = set()
        while key in aliases and key not in seen:
            seen.add(key)
            key = aliases[key]
        return min(seen | {key}) if key in seen else key

    shared = {}
    for key, item in entries.items():
        group = shared.setdefault(shared_identity(key), {
            "exists": False, "required": False, "expected_bytes": 0
        })
        group["exists"] |= item["exists"]
        group["required"] |= item["required"]
        group["expected_bytes"] = max(group["expected_bytes"], item["expected_bytes"])

    unique_files = set()
    disk_bytes = 0
    found = missing = stale = optional = folder_count = 0
    for item in entries.values():
        if not item["required"]:
            optional += 1
        if item["exists"]:
            found += 1
            if item["directory"]:
                folder_count += 1
                continue
            identity = os.path.normcase(os.path.realpath(item["path"]))
            if identity not in unique_files:
                unique_files.add(identity)
                disk_bytes += item["bytes"]
        elif item["configured"]:
            stale += 1
        else:
            missing += 1
    required_remaining = sum(g["expected_bytes"] for g in shared.values()
                             if not g["exists"] and g["required"])
    optional_remaining = sum(g["expected_bytes"] for g in shared.values()
                             if not g["exists"] and not g["required"])
    return {"files": entries, "summary": {
        "count": len(entries), "found": found, "missing": missing, "stale": stale,
        "optional": optional, "disk_bytes": disk_bytes, "folder_count": folder_count,
        "remaining_required_bytes": required_remaining,
        "remaining_optional_bytes": optional_remaining,
    }}
