"""Real local-git integration tests for upstream three-way merging and safety gates."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upstream_sync.py"


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], check=False,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def write(repo, filename, body):
    path = repo / filename
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(body, encoding="utf-8")
    git(repo, "add", filename)
    git(repo, "commit", "-m", f"Update {filename}")


@pytest.fixture
def repos(tmp_path):
    upstream = tmp_path / "upstream"
    fork = tmp_path / "fork"
    upstream.mkdir()
    git(upstream, "init", "-b", "master")
    git(upstream, "config", "user.name", "Test")
    git(upstream, "config", "user.email", "test@example.com")
    write(upstream, "shared.py", "VERSION=1\n")
    git(tmp_path, "clone", str(upstream), str(fork))
    git(fork, "config", "user.name", "Test")
    git(fork, "config", "user.email", "test@example.com")
    return upstream, fork


def sync(upstream, fork, *args):
    result = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(fork),
                             "--upstream", str(upstream), *args], text=True,
                            capture_output=True, check=False)
    data = json.loads(result.stdout.strip()) if result.stdout.strip().startswith("{") else {}
    return result, data


def test_prepare_merge_is_reviewable_preserves_fork_code_and_is_idempotent(repos):
    upstream, fork = repos
    write(fork, "fork_only.py", "FORK=True\n")
    old_master = git(fork, "rev-parse", "master")
    write(upstream, "upstream_only.py", "UPSTREAM=True\n")
    result, data = sync(upstream, fork, "--preserve", "fork_only.py")
    assert result.returncode == 0, result.stderr
    assert data["status"] == "prepared"
    assert data["branch"].startswith("sync/upstream-")
    assert git(fork, "rev-parse", "master") == old_master
    assert git(fork, "show", "HEAD:fork_only.py") == "FORK=True"
    assert git(fork, "show", "HEAD:upstream_only.py") == "UPSTREAM=True"
    assert len(git(fork, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3
    result, existing = sync(upstream, fork, "--preserve", "fork_only.py")
    assert result.returncode == 0
    assert existing["status"] == "existing"
    assert existing["branch"] == data["branch"]
    git(fork, "switch", "master")
    git(fork, "merge", "--ff-only", data["branch"])
    result, noop = sync(upstream, fork)
    assert result.returncode == 0
    assert noop["status"] == "up_to_date"


def test_conflict_aborts_without_touching_master(repos):
    upstream, fork = repos
    write(fork, "shared.py", "VERSION='fork'\n")
    old_master = git(fork, "rev-parse", "master")
    write(upstream, "shared.py", "VERSION='upstream'\n")
    result, data = sync(upstream, fork)
    assert result.returncode == 2, result.stderr
    assert data["status"] == "conflict"
    assert data["conflicts"] == "shared.py"
    assert git(fork, "rev-parse", "master") == old_master
    assert git(fork, "symbolic-ref", "--short", "HEAD") == "master"
    assert git(fork, "status", "--porcelain") == ""
    assert not (fork / ".git" / "MERGE_HEAD").exists()


def test_fork_only_content_cannot_disappear_silently(repos):
    upstream, fork = repos
    write(upstream, "new.py", "x=1\n")
    result, data = sync(upstream, fork, "--preserve", "fork/required.py")
    assert result.returncode == 2, result.stderr
    assert data["status"] == "missing_fork_files"
    assert data["conflicts"] == "fork/required.py"
    assert git(fork, "symbolic-ref", "--short", "HEAD") == "master"


def test_dirty_workspace_fails_closed(repos):
    upstream, fork = repos
    (fork / "dirty.txt").write_text("not committed")
    result, _data = sync(upstream, fork)
    assert result.returncode == 1
    assert "dirty working tree" in result.stderr


def test_remote_branch_is_reused_not_rewritten(repos, tmp_path):
    upstream, fork = repos
    published = tmp_path / "published.git"
    git(tmp_path, "init", "--bare", str(published))
    git(fork, "remote", "set-url", "origin", str(published))
    write(fork, "fork_only.py", "FORK=True\n")
    write(upstream, "upstream_only.py", "UPSTREAM=True\n")
    result, prepared = sync(upstream, fork)
    assert result.returncode == 0
    assert prepared["status"] == "prepared"
    git(fork, "push", "origin", f"HEAD:refs/heads/{prepared['branch']}")
    git(fork, "switch", "master")
    git(fork, "branch", "-D", prepared["branch"])
    result, existing = sync(upstream, fork)
    assert result.returncode == 0, result.stderr
    assert existing["status"] == "existing_remote"
    assert existing["branch"] == prepared["branch"]
    assert git(fork, "symbolic-ref", "--short", "HEAD") == "master"


def published_then_master_moves(upstream, fork, tmp_path, filename, body):
    """Publish a prepared sync branch, then commit `filename` to master after it."""
    published = tmp_path / "published.git"
    git(tmp_path, "init", "--bare", str(published))
    git(fork, "remote", "set-url", "origin", str(published))
    write(upstream, "upstream_only.py", "UPSTREAM=True\n")
    result, prepared = sync(upstream, fork)
    assert result.returncode == 0, result.stderr
    assert prepared["status"] == "prepared"
    git(fork, "push", "origin", f"HEAD:refs/heads/{prepared['branch']}")
    git(fork, "switch", "master")
    git(fork, "branch", "-D", prepared["branch"])
    write(fork, filename, body)
    return published, prepared


def test_published_branch_behind_master_is_brought_up_to_date_not_published_stale(repos, tmp_path):
    upstream, fork = repos
    published, prepared = published_then_master_moves(upstream, fork, tmp_path, "fork_later.py", "LATER=True\n")
    master = git(fork, "rev-parse", "master")
    result, refreshed = sync(upstream, fork)
    assert result.returncode == 0, result.stderr
    assert refreshed["status"] == "prepared"
    assert refreshed["branch"] == prepared["branch"]
    assert refreshed["base_sha"] == master
    assert refreshed["previous_sha"] == prepared["merge_sha"]
    head = refreshed["merge_sha"]
    assert git(fork, "rev-parse", "HEAD") == head
    # current master, the upstream commit and the published tip are all in it: pushing it is a fast-forward
    for ancestor in (master, refreshed["upstream_sha"], prepared["merge_sha"]):
        git(fork, "merge-base", "--is-ancestor", ancestor, head)
    assert git(fork, "show", f"{head}:fork_later.py") == "LATER=True"
    assert git(fork, "show", f"{head}:upstream_only.py") == "UPSTREAM=True"
    assert git(fork, "rev-parse", "master") == master
    assert git(published, "rev-parse", f"refs/heads/{prepared['branch']}") == prepared["merge_sha"]


def test_published_branch_that_cannot_take_master_is_a_conflict_left_unpublished(repos, tmp_path):
    upstream, fork = repos
    published, prepared = published_then_master_moves(upstream, fork, tmp_path, "upstream_only.py", "FORK=True\n")
    master = git(fork, "rev-parse", "master")
    result, data = sync(upstream, fork)
    assert result.returncode == 2, result.stderr
    assert data["status"] == "conflict"
    assert data["conflicts"] == "upstream_only.py"
    assert git(fork, "symbolic-ref", "--short", "HEAD") == "master"
    assert git(fork, "rev-parse", "master") == master
    assert git(fork, "status", "--porcelain") == ""
    assert git(published, "rev-parse", f"refs/heads/{prepared['branch']}") == prepared["merge_sha"]
