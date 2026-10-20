#!/usr/bin/env python3
"""Build a deterministic, balanced held-out camera-control evaluation manifest.

The manifest contains both the GT video (for ViPE) and the commanded c2w poses
(for H3 generation).  It deliberately does not use the command poses as the
metric target: the paper estimates poses from both videos with the same ViPE.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def absolutize(record: dict, base: Path) -> dict:
    value = dict(record)
    for key in ("video", "input_audio", "camera_poses"):
        if key in value:
            value[key] = str((base / value[key]).resolve())
    references = []
    for reference in value.get("references", []):
        reference = dict(reference)
        for key in ("image", "video"):
            if key in reference:
                reference[key] = str((base / reference[key]).resolve())
        references.append(reference)
    value["references"] = references
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.per_source <= 0:
        raise ValueError("--per-source must be positive")

    rows = []
    for source in ("1324-spatialvid", "2800-waymo"):
        metadata = json.loads((args.data_root / source / "val_metadata.json").read_text())
        if len(metadata) < args.per_source:
            raise ValueError(f"{source} has only {len(metadata)} validation records")
        chosen = random.Random(f"{args.seed}:{source}").sample(metadata, args.per_source)
        for index, record in enumerate(chosen):
            record = absolutize(record, args.data_root / source)
            record["id"] = f"{source}-{index:04d}"
            record["source"] = source
            rows.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} fixed evaluation records to {args.output}")


if __name__ == "__main__":
    main()
