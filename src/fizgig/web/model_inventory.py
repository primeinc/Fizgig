"""Cheap model-path inventory for Preferences. Read the same family descriptions as the form.

The configured path is authoritative. File sizes are actual logical bytes on
this computer; estimates for *missing* weights come from ModelFile.size_gb.
No searches, hashing, model loading, or downloads occur on this endpoint.
"""
import os


GIB = 1024 ** 3


def inventory(prefs, families, repo_dir, key_resolver=None):
    """Return keyed status and deduplicated totals for every Preferences model file."""
    entries = {}
    for family in families:
        known = {m.pref_key: m for m in family.model_files}
        # The form may expose additional option-specific model paths that do not
        # have a ModelFile entry. Check them too rather than marking them missing.
        keys = key_resolver(family) if key_resolver else known
        for key in keys:
            model = known.get(key)
            original = str(prefs.get(key) or "").strip()
            path = os.path.abspath(original if os.path.isabs(original)
                                   else os.path.join(repo_dir, original)) if original else ""
            try:
                size = os.path.getsize(path) if path and os.path.isfile(path) else 0
                exists = bool(path) and os.path.isfile(path)
            except OSError:
                size, exists = 0, False
            nominal = max(0, round(float(model.size_gb or 0) * GIB)) if model else 0
            previous = entries.get(key)
            if previous is None:
                entries[key] = {"exists": exists, "configured": bool(original),
                                "bytes": size if exists else 0, "expected_bytes": nominal,
                                "required": bool(model.required) if model else False,
                                "path": path if exists else ""}
            else:
                # The same preference may appear in several families: count it once.
                previous["required"] |= bool(model.required) if model else False
                previous["expected_bytes"] = max(previous["expected_bytes"], nominal)

    seen_files = set()
    disk_bytes = 0
    required_remaining = 0
    optional_remaining = 0
    found = missing = stale = optional = 0
    for item in entries.values():
        if not item["required"]:
            optional += 1
        if item["exists"]:
            found += 1
            identity = os.path.normcase(os.path.realpath(item["path"]))
            if identity not in seen_files:
                seen_files.add(identity)
                disk_bytes += item["bytes"]
        else:
            if item["configured"]:
                stale += 1
            else:
                missing += 1
            if item["required"]:
                required_remaining += item["expected_bytes"]
            else:
                optional_remaining += item["expected_bytes"]
    return {"files": entries, "summary": {
        "count": len(entries), "found": found, "missing": missing, "stale": stale,
        "optional": optional, "disk_bytes": disk_bytes,
        "remaining_required_bytes": required_remaining,
        "remaining_optional_bytes": optional_remaining,
    }}
