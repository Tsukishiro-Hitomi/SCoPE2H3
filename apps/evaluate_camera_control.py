"""Aggregate SCoPE-paper camera metrics from paired ViPE pose archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scope2h3.camera_evaluation import aggregate_metrics, evaluate_vipe_archives, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL: id, target_pose, predicted_pose")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for line_number, line in enumerate(args.pairs.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        pair = json.loads(line)
        missing = {"id", "target_pose", "predicted_pose"} - set(pair)
        if missing:
            raise ValueError(f"{args.pairs}:{line_number}: missing {sorted(missing)}")
        result = evaluate_vipe_archives(pair["target_pose"], pair["predicted_pose"])
        records.append({"id": pair["id"], **result})
    report = {"protocol": "ViPE-on-GT vs ViPE-on-generated; frame-0 anchored; raw scale; RotErr in radians", "summary": aggregate_metrics(records), "records": records}
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
