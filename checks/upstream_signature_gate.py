"""Compare fork callable signatures against the actual immutable upstream Git tree.

This is not a compatibility requirement for agent-authored fork behavior. It
protects upstream entrypoints from accidental parameter loss while integrating
a downstream feature. A fake-green fixture is attacked in the paired test.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import subprocess
import sys


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], text=True, capture_output=True, check=False)


def callables(source: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(source)
    result = {}

    def collect(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix + node.name] = node
            elif isinstance(node, ast.ClassDef):
                collect(node.body, prefix + node.name + ".")

    collect(tree.body)
    return result


def signature_violations(upstream_source: str, fork_source: str) -> list[str]:
    upstream = callables(upstream_source)
    fork = callables(fork_source)
    errors = []
    for name, original in upstream.items():
        adapted = fork.get(name)
        if adapted is None:
            errors.append(f"{name}: upstream callable missing from fork")
            continue
        a, b = original.args, adapted.args
        if b.kwarg is None:
            original_keywords = {x.arg for x in a.args + a.kwonlyargs}
            adapted_keywords = {x.arg for x in b.args + b.kwonlyargs}
            for lost in sorted(original_keywords - adapted_keywords):
                errors.append(f"{name}: upstream keyword parameter lost: {lost}")
        if b.vararg is None and len(b.posonlyargs) < len(a.posonlyargs):
            errors.append(f"{name}: upstream positional-only arity reduced")
    return errors


def check(upstream_ref: str, root: pathlib.Path) -> int:
    changed = 0
    problems = []
    ls = git("ls-tree", "-r", "--name-only", upstream_ref, "--", "src/fizgig")
    if ls.returncode:
        print(ls.stderr, file=sys.stderr)
        return 2
    for path in ls.stdout.splitlines():
        if not path.endswith(".py"):
            continue
        local = root / path
        if not local.is_file():
            problems.append(f"{path}: upstream Python file missing from fork")
            continue
        upstream = git("show", f"{upstream_ref}:{path}")
        if upstream.returncode:
            problems.append(f"{path}: could not read upstream blob: {upstream.stderr.strip()}")
            continue
        fork_text = local.read_text(encoding="utf-8")
        if upstream.stdout == fork_text:
            continue
        changed += 1
        try:
            problems.extend(f"{path} :: {item}" for item in signature_violations(upstream.stdout, fork_text))
        except SyntaxError as error:
            problems.append(f"{path}: source parse failed: {error}")
    print(f"Upstream callable contract: {changed} changed Python files inspected, {len(problems)} violations (source {upstream_ref})")
    for item in problems:
        print("INCOMPATIBLE: " + item, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-ref", required=True)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()
    raise SystemExit(check(args.upstream_ref, args.root))
