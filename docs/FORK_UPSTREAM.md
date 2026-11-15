# Maintaining the Fizgig fork against upstream

This is a maintenance contract and runbook, not a second model registry.

## Source of truth

- Canonical upstream: [shootthesound/Fizgig](https://github.com/shootthesound/Fizgig), `master`.
- Fork: [primeinc/Fizgig](https://github.com/primeinc/Fizgig), `master`.
- Upstream's `FamilyDescription`, the schema, model-file metadata, training code and GUI behavior remain authoritative **unless a demonstrated fork feature explicitly differs**.
- Prior fork code and tests are evidence of behavior, not automatic justification for retaining it. Do not preserve an accidental workaround just because it exists.

## Confirmed downstream functionality

| Concern | Fork-owned code | Upstream integration surface | Proof |
| --- | --- | --- | --- |
| Model search and reuse from other local libraries | `model_sources.py`, `fork/model_resolver.py` | Public wrappers in `scripts/fetch_models.py` | `checks/test_model_discovery.py`, `checks/test_download_respects_prefs.py` |
| ComfyUI/SwarmUI model download destinations | `model_destinations.py` | `fetch_models.py`, `web/server.py` | `checks/test_model_destinations.py`, `checks/test_download_prefs_handoff.py` |
| Model inventory and one-family-at-a-time model UI | `web/model_inventory.py`, `web/static/fork/model-library.js` | One Preferences render seam and `web/server.py` | `checks/model_library_contract.cjs`, `checks/test_h3_fork_integration.py` |
| Folder-scoped automatic caption replacements | `caption_replace.py`, `web/caption_rule_store.py`, `web/static/fork/caption-rules.js` | Caption worker and train CLI argument | `checks/test_caption_saved_rules.py`, `checks/caption_rules_contract.cjs`, `checks/test_train_cli_contract.py` |
| Fork-only browser presentation | `web/static/fork/*.css` | Explicit, constrained `/fork/` static URL namespace | Recursive JS gate and UI contracts |

**Intentional policy difference:** a valid, explicitly configured model path is not replaced when upstream model metadata advertises a renamed or alternate build. Users clear the model-path preference to authorize a different downloaded build. This is a downstream decision and must remain visible in reviews, not silently attributed to upstream.

## Update lifecycle

1. `.github/workflows/upstream-sync.yml` checks upstream every day and can also be started manually. It runs local Git merge/conflict/idempotence tests before attempting synchronization.
2. `scripts/upstream_sync.py` fetches the actual upstream `master` revision, checks whether that commit is already an ancestor, and creates `sync/upstream-<sha>` from the current fork `master` only when needed.
3. The merge must be a real two-parent merge, retain the fork's declared modules, and have no unresolved conflicts. Conflicts, unrelated histories and missing fork modules **fail closed**.
4. Before publishing a newly prepared sync branch, the automation runs the canonical CPU checks, Python compilation and recursive browser-JavaScript parsing. Only then may it push and open a review PR.
5. It **never merges automatically into `master`**. A real upstream change can affect GPU behavior that CPU checks cannot demonstrate. The submitted PR needs the normal acceptance process.
6. GitHub's `GITHUB_TOKEN` does not reliably start ordinary CI for a PR it creates. The workflow explicitly dispatches `checks.yml` on the sync branch using `workflow_dispatch` (first-party GitHub exception).
7. The deterministic branch name and remote-branch check prevent repeatedly creating a new merge commit for the same upstream revision. If a prior run produced the branch but not the PR, the next run can recover.
8. A blocked run stays visible as a failed GitHub Actions job; it must not report an unverified success. A GitHub repository must permit GitHub Actions to create pull requests for the automation to open them using the default token. A repository restriction is an actual operational blocker, not a reason to bypass authorization.

## Canonical gates

```bash
python -m compileall -q src
python checks/js_parse_gate.py
python -m pytest checks -q
```

`checks/test_upstream_sync.py` uses real isolated Git repositories to prove the merge path, a conflicting source file, preservation failure and repeat-run behavior. `checks/test_js_parse_gate.py` proves a malformed nested fork script is rejected. `checks/test_train_cli_contract.py` compares the CLI's real training call with the real `train_family` signature, including a deliberately broken signature.

## Closure boundary

Green CPU tests do not prove GPU-backed inference, caption model downloads, on-device training, or visual correctness. Those capabilities require explicit hardware-backed acceptance separately. An upstream release should not be described as completely supported until the claimed hardware behaviors are actually tested.

Upstream updates may still require a domain decision when upstream and the fork intentionally implement different behaviors. The automated sync removes routine manual replay, not the need to resolve genuine semantic conflicts.
