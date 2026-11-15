"""Fork-owned local model reuse and destination-aware fetch implementation.

Called through stable `fizgig.scripts.fetch_models` public wrappers. The caller
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

def find_weight(w, models_dir, prefs, scan_cache=None, repo_dir=None, allow_alternates=False, *, core):
    """Resolve without downloading or mutating prefs, honouring existing explicit selections.

    The fork's explicit-path policy takes precedence over upstream switch_from/replaces
    hints. A valid configured file stays selected until the user clears the row.
    Discovery can find alternate builds only when a model is otherwise unresolved.
    """
    current = core._configured_file(prefs.get(w.pref_key), repo_dir)
    if current:
        return current
    other_key = getattr(w, "same_as", "")
    other = str(prefs.get(other_key) or "").strip() if other_key else ""
    if core._configured_file(other, repo_dir):
        return core._configured_file(other, repo_dir)
    min_bytes = int(w.gb * 0.8 * 1024 ** 3)
    found = core.model_sources.resolve_weight(w, prefs.get("model_search_roots", []), models_dir,
                                         core._valid_safetensors, min_bytes, cache=scan_cache)
    if found is core.model_sources.AMBIGUOUS or found:
        return found
    # Search Locally can reuse an advertised alternative build even if the
    # model does not offer a downloader build-choice dropdown. Bulk downloads
    # deliberately do not switch builds unless explicitly selected.
    if allow_alternates and getattr(w, "alt", None):
        alt = w.alt
        alt_found = core.model_sources.resolve_weight(alt, prefs.get("model_search_roots", []), models_dir,
                          core._valid_safetensors, int(alt.gb * 0.8 * 1024 ** 3), cache=scan_cache)
        if alt_found is core.model_sources.AMBIGUOUS or alt_found:
            return alt_found
    # The chosen output library is also a source, but user roots and the HF cache
    # retain precedence. A missing external destination must not break local discovery.
    try:
        _download_parent, saved_path = core.model_destinations.destination(w, models_dir, prefs)
    except ValueError:
        return None
    if os.path.isfile(saved_path) and core._valid_safetensors(saved_path, min_bytes):
        return saved_path
    return None

def fetch_weight(w, models_dir, prefs, token=None, log=print, dry_run=False, scan_cache=None, repo_dir=None, *, core):
    """Download one weight if needed and point its pref at it. Returns True if usable."""
    # 80% of the nominal size — generous enough for rounding, tight enough to catch a truncation.
    min_bytes = int(w.gb * 0.8 * 1024 ** 3)

    # A pref the user already set, pointing at a file that exists, is LEFT ALONE — existence is
    # the only test. Deliberately not the size check below: that exists to catch a transfer WE
    # just made being truncated, and applying it here would second-guess legitimate choices.
    # Plenty are smaller than the file this manifest names — the fp8_scaled RAW DiT instead of
    # bf16, Klein's fp8mixed text encoder instead of the full one, a community fine-tune — and
    # every one of those would have been "too small", silently re-downloaded, and the user's
    # path overwritten with ours.
    current = core._configured_file(prefs.get(w.pref_key), repo_dir)
    if current:
        log(f"  [keep] {w.pref_key} -> {current} (existing configured model; no download)")
        return True

    # the same file set for another model (any name: the user's own copy, e.g. ComfyUI's renamed VAE): linked, not fetched
    same_as = getattr(w, "same_as", "")
    other = str(prefs.get(same_as) or "").strip() if same_as else ""
    if core._configured_file(other, repo_dir):
        log(f"  [have] {w.filename} — already set for another model ({os.path.basename(other)}), linking into "
            "Preferences")
        prefs[w.pref_key] = str(prefs.get(same_as))
        return True

    found = core.find_weight(w, models_dir, prefs, scan_cache=scan_cache, repo_dir=repo_dir)
    if found is core.model_sources.AMBIGUOUS:
        log(f"  [ambiguous] {w.filename} — equally strong local files; choose the correct file in Preferences.")
        return False
    if found:
        log(f"  [have] {w.filename} — {found}, linking into Preferences")
        prefs[w.pref_key] = found
        return True

    # A populated Preferences row is a user selection. If it no longer exists and
    # discovery found nothing, do not silently turn it into a multi-GB download.
    # Clearing that row explicitly is how the user authorises a fresh download.
    configured = str(prefs.get(w.pref_key) or "").strip()
    if configured:
        log(f"  [stale] {w.pref_key}: saved file not found: {configured}")
        log("          Searched your saved model directories and caches; no match. "
            "Correct or clear the Preferences path before downloading.")
        return False

    try:
        download_parent, dest = core.model_destinations.destination(w, models_dir, prefs)
    except ValueError as error:
        log(f"  [fail] {w.filename}: {error}")
        return False
    if dry_run:
        log(f"  [get]  {w.filename} (~{w.gb:g} GB) — {w.note}")
        return True
    if w.gated and not (token or os.environ.get("HF_TOKEN", "").strip()):
        log(f"  [gated] {w.filename} is missing locally; {w.repo} needs a Hugging Face token.")
        return False

    log(f"  [get]  {w.filename} (~{w.gb:g} GB) — {w.note}")
    try:
        os.makedirs(download_parent, exist_ok=True)
        if core.EMIT_PROGRESS:
            got = core._download_with_progress(w, download_parent, token, log)
        else:
            from fizgig.utils import hf_public
            got = hf_public.download(repo_id=w.repo, filename=w.path_in_repo, local_dir=download_parent,
                                     token=token or os.environ.get("HF_TOKEN") or None, log=log)
    except Exception as e:
        name = type(e).__name__
        from fizgig.utils import hf_public
        if isinstance(e, hf_public.LoginExpired):
            log(f"  [login] {w.filename}: {e}")
        elif "Gated" in name or "401" in str(e) or "403" in str(e):
            log(f"  [gated] {w.repo} needs you to accept its licence first:")
            log(f"          https://huggingface.co/{w.repo}")
            log(f"          Then re-run with --token hf_... (or set HF_TOKEN).")
        else:
            log(f"  [fail] {w.filename}: {name}: {e}")
        return False

    # Validate the downloaded artifact before moving it into another application's library.
    got = os.path.abspath(got)
    if not core._valid_safetensors(got, min_bytes):
        log(f"  [fail] {w.filename} downloaded but failed verification — leaving it for a re-run")
        return False
    if os.path.normcase(got) != os.path.normcase(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(got, dest)
        d = os.path.dirname(got)
        while os.path.normcase(d) != os.path.normcase(download_parent):
            try:
                os.rmdir(d)
            except OSError:
                break
            d = os.path.dirname(d)

    prefs[w.pref_key] = dest
    log(f"  [done] {w.filename}")
    return True
