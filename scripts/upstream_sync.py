"""Prepare a non-destructive, reviewable three-way upstream sync branch.

No history rewrite, working-tree changes to the base branch, or implicit merge to
master. Exit statuses are emitted to GITHUB_OUTPUT for the scheduled workflow.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", "-C", str(repo), *args], text=True,
                            capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result


def emit(data: dict[str, str]) -> None:
    print(json.dumps(data, sort_keys=True))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as file:
            for key, value in data.items():
                file.write(f"{key}={value}\n")


def prepare(repo: Path, upstream: str, base: str, preserve: tuple[str, ...]) -> dict[str, str]:
    repo = repo.resolve()
    if not git(repo, "status", "--porcelain").stdout.strip() == "":
        raise RuntimeError("Refusing to sync a dirty working tree")
    base_sha = git(repo, "rev-parse", f"refs/heads/{base}").stdout.strip()
    git(repo, "fetch", "--no-tags", upstream, "refs/heads/master")
    upstream_sha = git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()
    if git(repo, "merge-base", "--is-ancestor", upstream_sha, base_sha, check=False).returncode == 0:
        return {"status": "up_to_date", "upstream_sha": upstream_sha, "base_sha": base_sha}
    common = git(repo, "merge-base", base_sha, upstream_sha, check=False)
    if common.returncode != 0:
        raise RuntimeError("Refusing to merge an unrelated upstream history")
    branch = f"sync/upstream-{upstream_sha[:12]}"
    remote = git(repo, "ls-remote", "--heads", "origin", f"refs/heads/{branch}", check=False)
    if remote.returncode == 0 and remote.stdout.strip():
        # A previous scheduled run already published the deterministic branch.
        # Never force-push or rebuild it with a new, non-FF merge commit.
        git(repo, "fetch", "--no-tags", "origin", f"refs/heads/{branch}")
        remote_sha = git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()
        if git(repo, "merge-base", "--is-ancestor", upstream_sha, remote_sha, check=False).returncode:
            raise RuntimeError(f"Published branch {branch} does not contain {upstream_sha}")
        return {"status": "existing_remote", "branch": branch,
                "upstream_sha": upstream_sha, "base_sha": base_sha}
    existing = git(repo, "rev-parse", "--verify", f"refs/heads/{branch}", check=False)
    if existing.returncode == 0:
        known = git(repo, "merge-base", "--is-ancestor", upstream_sha, existing.stdout.strip(), check=False)
        if known.returncode != 0:
            raise RuntimeError(f"Existing branch {branch} does not contain {upstream_sha}")
        return {"status": "existing", "branch": branch, "upstream_sha": upstream_sha, "base_sha": base_sha}
    git(repo, "switch", "-c", branch, base)
    try:
        merge = git(repo, "-c", "user.name=Fizgig Upstream Sync",
                    "-c", "user.email=upstream-sync@users.noreply.github.com",
                    "merge", "--no-ff", "--no-edit", "-m", f"Sync upstream {upstream_sha}",
                    upstream_sha, check=False)
        if merge.returncode:
            unmerged = git(repo, "diff", "--name-only", "--diff-filter=U").stdout.strip()
            git(repo, "merge", "--abort")
            git(repo, "switch", base)
            git(repo, "branch", "-D", branch)
            return {"status": "conflict", "branch": branch, "upstream_sha": upstream_sha,
                    "base_sha": base_sha, "conflicts": ",".join(unmerged.splitlines())}
        missing = [p for p in preserve if git(repo, "cat-file", "-e", f"HEAD:{p}", check=False).returncode]
        if missing:
            git(repo, "switch", base)
            git(repo, "branch", "-D", branch)
            return {"status": "missing_fork_files", "upstream_sha": upstream_sha,
                    "base_sha": base_sha, "conflicts": ",".join(missing)}
        return {"status": "prepared", "branch": branch, "upstream_sha": upstream_sha,
                "base_sha": base_sha, "merge_sha": git(repo, "rev-parse", "HEAD").stdout.strip()}
    except Exception:
        # The caller will fail loudly; don't silently reset a partially failed merge.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--upstream", default="https://github.com/shootthesound/Fizgig.git")
    parser.add_argument("--base", default="master")
    parser.add_argument("--preserve", action="append", default=[])
    args = parser.parse_args()
    try:
        result = prepare(args.repo, args.upstream, args.base, tuple(args.preserve))
    except Exception as exc:
        print(f"upstream sync failed: {exc}", file=sys.stderr)
        return 1
    emit(result)
    return 0 if result["status"] in {"prepared", "existing", "existing_remote", "up_to_date"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
