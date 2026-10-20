#!/usr/bin/env python3
"""Create pair records for raw-scale metric aggregation after two ViPE passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-vipe-root", type=Path, required=True)
    parser.add_argument("--generated-vipe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        gt_name = Path(record["video"]).stem
        pair = {
            "id": record["id"],
            "target_pose": str(args.gt_vipe_root / "pose" / f"{gt_name}.npz"),
            "predicted_pose": str(args.generated_vipe_root / "pose" / f"{record['id']}.npz"),
        }
        absent = [path for path in (pair["target_pose"], pair["predicted_pose"]) if not Path(path).is_file()]
        if absent:
            raise FileNotFoundError(f"{record['id']}: missing ViPE output: {absent}")
        pairs.append(pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(pair) + "\n" for pair in pairs), encoding="utf-8")
    print(f"wrote {len(pairs)} ViPE metric pairs to {args.output}")


if __name__ == "__main__":
    main()
