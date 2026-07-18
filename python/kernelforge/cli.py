from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import traffic_estimate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze KernelForge memory traffic")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    payload = traffic_estimate(args.rows, args.hidden).as_dict()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0

