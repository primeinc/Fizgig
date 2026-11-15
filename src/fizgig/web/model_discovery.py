"""Fork-only local model discovery worker.

The upstream server supplies its live module as core, so tests and runtime
continue to use the same prefs, locks, thread scheduler, and fetch state.
Based on primeinc/Fizgig@1dbd2aabab8498254886b064c04f2235c33dd1c6 :: web/server.py#_start_model_search.
"""
import os

STATE = {"state": "idle", "family": "", "processed": 0, "total": 0,
                 "found": 0, "kept": 0, "missing": 0, "ambiguous": 0, "results": [], "error": ""}


def status(core):
    with core._lock:
        return {**STATE, "results": list(STATE["results"])}


def start(family="", choices=None, *, core):
    """Search local files on a worker without downloading; preserve explicit paths and edits."""
    from fizgig.scripts import fetch_models
    family = str(family or "").strip()
    if family and family not in fetch_models.FAMILIES:
        raise ValueError("Unknown model family.")
    allowed = {f.pref_key for d in core.training_families() for f in d.model_files if f.fetch_choice}
    choices = {str(k): v for k, v in (choices or {}).items() if k in allowed and v in ("main", "alt")}
    weights = list(fetch_models.discovery_candidates((family,) if family else None, choices=choices))
    with core._lock:
        if STATE["state"] == "running":
            raise ValueError("A local model search is already running.")
        if core._fetch["proc"] is not None and core._fetch["proc"].poll() is None:
            raise ValueError("A model download is running; search after it finishes.")
        STATE.update(state="running", family=family, processed=0, total=len(weights),
                             found=0, kept=0, missing=0, ambiguous=0, results=[], error="")

    def work():
        prefs = core.load_prefs()
        updates = {}
        scan_cache = {}  # walk each configured model library at most once per search
        try:
            for w in weights:
                result = fetch_models.find_weight(w, core.user_data.MODELS_DIR, prefs, scan_cache=scan_cache,
                                                  allow_alternates=True)
                current = str(prefs.get(w.pref_key) or "").strip()
                if result is fetch_models.model_sources.AMBIGUOUS:
                    status, path = "ambiguous", ""
                elif not result:
                    status, path = "missing", ""
                elif current and os.path.isfile(core.pref_path(prefs, w.pref_key)) and not (
                        w.switch_from and os.path.basename(current) == w.switch_from) and (
                        os.path.basename(current) not in w.replaces):
                    status, path = "kept", current
                else:
                    status, path = "found", os.path.abspath(result)
                    updates[w.pref_key] = path
                with core._lock:
                    STATE[status] += 1
                    STATE["processed"] += 1
                    STATE["results"].append({"key": w.pref_key, "status": status, "path": path})
            if updates:
                with core._lock:
                    latest = core.load_prefs()
                    # A user who changed a path mid-search has final say.
                    safe = {k: v for k, v in updates.items()
                            if str(latest.get(k) or "").strip() == str(prefs.get(k) or "").strip()}
                    if safe:
                        latest.update(safe)
                        core._write_json(core.PREFS_FILE, latest)
            with core._lock:
                STATE["state"] = "done"
        except Exception as e:
            core.logger.exception("Local model discovery")
            with core._lock:
                STATE.update(state="failed", error=f"{type(e).__name__}: {e}")

    core.threading.Thread(target=work, name="fizgig-model-discovery", daemon=True).start()
