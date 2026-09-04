#!/usr/bin/env python
"""Fail if pytest collects fewer tests than the suite is known to contain.

The problem this solves. `tests/test_baselines.py` opens with a module-level
`pytest.importorskip("torch")`, so on a runner without torch the entire file --
32 collected cases, including the equivalence assertions and their negative
controls -- vanishes from the run. pytest still exits 0. Before this guard,
ci.yml never installed torch, so it silently collected 32 of 64 cases and
reported success; a `-q` summary line reading "31 passed" looks indistinguishable
from a healthy run unless you already know the real number.

A skip is a legitimate outcome -- torch is a heavy optional dependency and the
CPU-only lane should not pay for it. What is not legitimate is a skip nobody
declared. So this script pins the inventory per module and per profile:

    core  what ci.yml can collect with the `dev` extra alone. test_baselines.py
          is expected to be absent, and every other module must be complete.
    full  what baseline-comparison.yml must collect, with CPU torch installed.
          test_baselines.py is required to be present and whole; if the torch
          install silently fails, this profile fails rather than degrading.

Deliberate choice: ci.yml is NOT given torch. It stays the fast lane on Python
3.10/3.12, and the torch tier is owned by baseline-comparison.yml, which installs
the CPU wheel and runs the `full` profile on 3.11/3.12. So the 32 torch cases do
run in CI -- just in the job that declares the dependency. The floor here is what
each job actually collects, not an aspiration.

When you add or remove tests, update EXPECTED below in the same commit. That is
the intended friction: it makes a change in collected count a reviewed decision
instead of an invisible one.

Usage:
    python scripts/check_test_inventory.py --profile core
    python scripts/check_test_inventory.py --profile full
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# module -> number of collected cases (parametrization expanded).
# Verified with `pytest --collect-only -q` at the time of writing.
EXPECTED: dict[str, int] = {
    "tests/test_baselines.py": 32,
    "tests/test_compare.py": 9,
    "tests/test_matrix.py": 3,
    "tests/test_model.py": 3,
    "tests/test_triton_codegen_check.py": 17,
}

# Modules that disappear entirely without torch, via a module-level importorskip.
TORCH_ONLY = frozenset({"tests/test_baselines.py"})

NODE_ID = re.compile(r"^(tests[/\\][\w_]+\.py)::")


def collect(root: Path) -> dict[str, int]:
    """Count collected cases per module via `pytest --collect-only`.

    `-o addopts=` clears the project's `-q` from pyproject.toml so the output is
    node ids, which are stable to parse, rather than the per-file summary that
    `-q` switches to.
    """

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        print("pytest collection itself failed:", file=sys.stderr)
        print(completed.stdout[-4000:], file=sys.stderr)
        print(completed.stderr[-2000:], file=sys.stderr)
        raise SystemExit(2)
    counts: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        match = NODE_ID.match(line.strip())
        if match:
            module = match.group(1).replace("\\", "/")
            counts[module] = counts.get(module, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "full"), required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    root = Path(args.root)
    actual = collect(root)

    if args.profile == "full":
        expected = dict(EXPECTED)
    else:
        expected = {k: v for k, v in EXPECTED.items() if k not in TORCH_ONLY}

    failures: list[str] = []
    for module, count in sorted(expected.items()):
        got = actual.get(module, 0)
        if got < count:
            failures.append(
                f"{module}: collected {got}, expected {count} "
                f"({count - got} case(s) missing -- an import guard or a deleted "
                f"test, not a passing run)"
            )
        elif got > count:
            failures.append(
                f"{module}: collected {got}, expected {count} -- tests were added; "
                f"raise the floor in scripts/check_test_inventory.py"
            )

    # On the core profile the torch modules must be absent for the *declared*
    # reason. If they show up, torch leaked into the lean lane and the profile no
    # longer describes reality.
    if args.profile == "core":
        for module in sorted(TORCH_ONLY):
            if actual.get(module):
                failures.append(
                    f"{module}: collected {actual[module]} on the 'core' profile, which "
                    f"expects it to skip. torch is now installed in this job -- switch it "
                    f"to --profile full so those cases are actually enforced."
                )

    for module in sorted(set(actual) - set(EXPECTED)):
        failures.append(
            f"{module}: collected {actual[module]} case(s) but is not in EXPECTED; "
            f"add it to scripts/check_test_inventory.py"
        )

    total = sum(actual.get(module, 0) for module in expected)
    if failures:
        print(f"Test collection does not match the '{args.profile}' inventory:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nA suite that collects fewer tests than expected still exits 0, which is "
            "exactly what this check exists to catch.",
            file=sys.stderr,
        )
        return 1

    print(f"profile '{args.profile}': collected {total} cases across {len(expected)} modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
