#!/usr/bin/env python3
"""Partition a JSONL evaluation manifest deterministically without overlap."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be positive")
    lines = [line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("manifest is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    groups = [[] for _ in range(args.shards)]
    for index, line in enumerate(lines):
        groups[index % args.shards].append(line)
    for shard, group in enumerate(groups):
        output = args.output_dir / f"manifest-{shard:02d}-of-{args.shards:02d}.jsonl"
        output.write_text("\n".join(group) + "\n", encoding="utf-8")
        print(f"{output}: {len(group)} records")


if __name__ == "__main__":
    main()
