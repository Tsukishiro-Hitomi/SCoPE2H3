#!/usr/bin/env python3
"""Keep camera-evaluation records whose generated videos exist on disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [record for record in records if (args.generated_root / f"{record['id']}.mp4").is_file()]
    if not selected:
        raise RuntimeError(f"no generated videos found under {args.generated_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record) + "\n" for record in selected), encoding="utf-8"
    )
    print(f"wrote {len(selected)}/{len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
