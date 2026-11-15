"""Fork-owned local model discovery and configured-path resolution.

Called through stable `fizgig.scripts.fetch_models` public resolver wrappers. The caller
passes its current module as `core` to preserve monkeypatchable dependencies
and Python `-m fizgig.scripts.fetch_models` (__main__) module identity.
The upstream FamilyDescription registry remains the canonical model catalog.

Intentional downstream divergence: a valid user-selected Preferences path is
never replaced based on a filename hint or a default build choice. Clearing
the row opts into discovery/download, which may use upstream variant metadata.
"""
from __future__ import annotations
import os


def configured_file(value, repo_dir):
    """A saved Preferences model path is authoritative if its file exists.

    Preferences may contain paths relative to Fizgig's repository; the downloader
    runs in a subprocess and must resolve them exactly as the web app does.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = raw if os.path.isabs(raw) else os.path.join(repo_dir, raw)
    return os.path.abspath(path) if os.path.isfile(path) else ""

def found_file_check(w, core):
    """The test a discovered file passes to be linked for `w`: a plausible .safetensors file whose header does not
    prove it holds another model's weights (ModelFile.signature, fetch_models.is_other_file)."""
    signature = getattr(w, "signature", ())
    return lambda path, min_bytes: (core._valid_safetensors(path, min_bytes)
                                    and not core.is_other_file(path, signature))


def find_weight(w, models_dir, prefs, scan_cache=None, repo_dir=None, allow_alternates=False, *, core):
    """Resolve without downloading or mutating prefs, honouring existing explicit selections.

    The fork's explicit-path policy takes precedence over upstream switch_from/replaces
    hints. A valid configured file stays selected until the user clears the row.
    Discovery can find alternate builds only when a model is otherwise unresolved, and
    never links a file whose header shows it is another model's.
    """
    current = core._configured_file(prefs.get(w.pref_key), repo_dir)
    if current:
        return current
    other_key = getattr(w, "same_as", "")
    other = str(prefs.get(other_key) or "").strip() if other_key else ""
    if core._configured_file(other, repo_dir):
        return core._configured_file(other, repo_dir)
    min_bytes = int(w.gb * 0.8 * 1024 ** 3)
    valid = found_file_check(w, core)
    found = core.model_sources.resolve_weight(w, prefs.get("model_search_roots", []), models_dir,
                                         valid, min_bytes, cache=scan_cache)
    if found is core.model_sources.AMBIGUOUS or found:
        return found
    # Search Locally can reuse an advertised alternative build even if the
    # model does not offer a downloader build-choice dropdown. Bulk downloads
    # deliberately do not switch builds unless explicitly selected.
    if allow_alternates and getattr(w, "alt", None):
        alt = w.alt
        alt_found = core.model_sources.resolve_weight(alt, prefs.get("model_search_roots", []), models_dir,
                          found_file_check(alt, core), int(alt.gb * 0.8 * 1024 ** 3), cache=scan_cache)
        if alt_found is core.model_sources.AMBIGUOUS or alt_found:
            return alt_found
    # The chosen output library is also a source, but user roots and the HF cache
    # retain precedence. A missing external destination must not break local discovery.
    try:
        _download_parent, saved_path = core.model_destinations.destination(w, models_dir, prefs)
    except ValueError:
        return None
    if os.path.isfile(saved_path) and valid(saved_path, min_bytes):
        return saved_path
    return None
