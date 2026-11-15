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
    # Sources are UTF-8 whatever the platform's locale code page is.
    return subprocess.run(["git", *args], text=True, encoding="utf-8", capture_output=True, check=False)


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


def optional_positions(args: ast.arguments) -> set[int]:
    """Indexes of the positional parameters that have a default (defaults fill the last ones)."""
    count = len(args.posonlyargs) + len(args.args)
    return set(range(count - len(args.defaults), count))


def call_breaks(a: ast.arguments, b: ast.arguments) -> list[str]:
    """How a call that binds to upstream parameters `a` fails, or binds otherwise, against fork parameters `b`."""
    problems = []
    a_pos = [*a.posonlyargs, *a.args]
    b_pos = [*b.posonlyargs, *b.args]
    a_opt, b_opt = optional_positions(a), optional_positions(b)
    b_by_keyword = {x.arg for x in b.args + b.kwonlyargs}
    b_kw_optional = {x.arg for x, default in zip(b.kwonlyargs, b.kw_defaults) if default is not None}

    # Each upstream positional slot still takes an argument by position, the same parameter, as optional as before.
    for i, p in enumerate(a_pos):
        if i >= len(b_pos):
            if b.vararg is None:
                problems.append(f"upstream positional parameter {i + 1} ({p.arg}) no longer accepted by position")
            continue
        q = b_pos[i]
        if i >= len(a.posonlyargs) and q.arg != p.arg:
            problems.append(f"upstream positional parameter {i + 1} ({p.arg}) is now {q.arg}")
        elif i in a_opt and i not in b_opt:
            problems.append(f"upstream optional parameter made required: {p.arg}")

    # Each upstream keyword is still taken by keyword, unless the fork takes **kwargs.
    if b.kwarg is None:
        for lost in sorted({x.arg for x in a.args + a.kwonlyargs} - b_by_keyword):
            problems.append(f"upstream keyword parameter lost: {lost}")

    # An upstream optional keyword-only parameter the fork still names stays optional.
    for x, default in zip(a.kwonlyargs, a.kw_defaults):
        if default is None or x.arg not in b_by_keyword:
            continue
        at = next((j for j, q in enumerate(b_pos) if q.arg == x.arg), None)
        if x.arg not in b_kw_optional and (at is None or at not in b_opt):
            problems.append(f"upstream optional parameter made required: {x.arg}")

    if a.vararg is not None and b.vararg is None:
        problems.append("upstream *args removed")
    if a.kwarg is not None and b.kwarg is None:
        problems.append("upstream **kwargs removed")

    # A parameter the fork requires that upstream never had fails every upstream call.
    a_names = {x.arg for x in a_pos + a.kwonlyargs}
    for j, q in enumerate(b_pos):
        if j >= len(a_pos) and j not in b_opt and q.arg not in a_names:
            problems.append(f"fork adds required parameter: {q.arg}")
    for x, default in zip(b.kwonlyargs, b.kw_defaults):
        if default is None and x.arg not in a_names:
            problems.append(f"fork adds required parameter: {x.arg}")
    return problems


def signature_violations(upstream_source: str, fork_source: str) -> list[str]:
    upstream = callables(upstream_source)
    fork = callables(fork_source)
    errors = []
    for name, original in upstream.items():
        adapted = fork.get(name)
        if adapted is None:
            errors.append(f"{name}: upstream callable missing from fork")
            continue
        errors.extend(f"{name}: {problem}" for problem in call_breaks(original.args, adapted.args))
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
