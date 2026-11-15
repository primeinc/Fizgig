"""The upstream server delegates fork discovery without duplicating mutable state."""
from fizgig.web import model_discovery, server


def test_fork_model_discovery_state_is_shared_with_upstream_server():
    assert server._model_search is model_discovery.STATE
    with server._lock:
        original = dict(server._model_search)
        try:
            server._model_search.update(state="idle", results=[{"key": "probe", "status": "found"}])
            status = server._model_search_status()
            assert status["state"] == "idle"
            assert status["results"][0]["key"] == "probe"
            status["results"].append({"key": "mutated_response"})
            assert len(server._model_search["results"]) == 1, "Status response leaked mutable state"
        finally:
            server._model_search.clear()
            server._model_search.update(original)


def test_public_facade_delegates_to_live_server_dependencies(monkeypatch):
    seen = {}

    def prove_core(family="", choices=None, *, core):
        seen["core"] = core
        seen["family"] = family
        seen["choices"] = choices
        return "delegate-hit"

    monkeypatch.setattr(model_discovery, "start", prove_core)
    assert server._start_model_search("new_future_family", {"dit": "alt"}) == "delegate-hit"
    assert seen == {"core": server, "family": "new_future_family", "choices": {"dit": "alt"}}
