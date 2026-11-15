"""Parse every shipped browser JavaScript file, including fork subdirectories.

Run via `python checks/js_parse_gate.py`. This is an explicit closure gate for
the fork-owned static modules introduced in PRs #13 and #14, which the
upstream `static/*.js` shell loop did not include.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "src" / "fizgig" / "web" / "static"


def check_all(root: Path) -> int:
    if not root.is_dir():
        print(f"JS gate: missing source directory: {root}", file=sys.stderr)
        return 1
    scripts = sorted(root.rglob("*.js"))
    if not scripts:
        print(f"JS gate: no JavaScript files under {root}", file=sys.stderr)
        return 1
    failures = 0
    for path in scripts:
        result = subprocess.run(["node", "--check", str(path)], text=True,
                                capture_output=True, check=False)
        if result.returncode:
            failures += 1
            print(f"JS syntax FAIL: {path}", file=sys.stderr)
            print(result.stderr or result.stdout, file=sys.stderr)
    print(f"JS gate: {len(scripts)} files checked, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    raise SystemExit(check_all(args.root))
