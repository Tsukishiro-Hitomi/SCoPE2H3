"""Create portable SVG charts and a summary from an H3/SCoPE training run."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


WIDTH, HEIGHT, PAD = 1100, 520, 70


def polyline_svg(points: list[tuple[float, float]], title: str, x_label: str, y_label: str) -> str:
    if not points:
        return ""
    xs, ys = zip(*points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    x_scale = lambda value: PAD + (value - x_min) / (x_max - x_min) * (WIDTH - 2 * PAD)
    y_scale = lambda value: HEIGHT - PAD - (value - y_min) / (y_max - y_min) * (HEIGHT - 2 * PAD)
    line = " ".join(f"{x_scale(x):.2f},{y_scale(y):.2f}" for x, y in points)
    esc = html.escape
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<rect width="100%" height="100%" fill="white"/>
<text x="{WIDTH / 2}" y="30" text-anchor="middle" font-size="22" font-family="sans-serif">{esc(title)}</text>
<line x1="{PAD}" y1="{HEIGHT - PAD}" x2="{WIDTH - PAD}" y2="{HEIGHT - PAD}" stroke="#333"/>
<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{HEIGHT - PAD}" stroke="#333"/>
<polyline points="{line}" fill="none" stroke="#1565c0" stroke-width="2"/>
<text x="{WIDTH / 2}" y="{HEIGHT - 20}" text-anchor="middle" font-size="16" font-family="sans-serif">{esc(x_label)} ({x_min:g}–{x_max:g})</text>
<text x="18" y="{HEIGHT / 2}" text-anchor="middle" font-size="16" font-family="sans-serif" transform="rotate(-90 18 {HEIGHT / 2})">{esc(y_label)} ({y_min:.4g}–{y_max:.4g})</text>
</svg>\n'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--elapsed-seconds", type=float, default=None)
    args = parser.parse_args()
    root = args.output_path
    losses: list[tuple[float, float]] = []
    loss_path = root / "loss.csv"
    if loss_path.is_file():
        with loss_path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("key") == "loss":
                    losses.append((float(row["step"]), float(row["value"])))

    profile_dir = root / "profiling"
    timing: dict = {}
    timing_path = profile_dir / "timing_summary.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    memory_records: list[dict] = []
    for path in sorted(profile_dir.glob("memory_rank*.jsonl")):
        with path.open(encoding="utf-8") as file:
            memory_records.extend(json.loads(line) for line in file if line.strip())

    max_allocated = max((float(row.get("peak_allocated_mib", 0)) for row in memory_records), default=None)
    max_reserved = max((float(row.get("peak_reserved_mib", 0)) for row in memory_records), default=None)
    memory_points = [
        (float(row["step"] or 0), float(row.get("peak_allocated_mib", 0)))
        for row in memory_records if row.get("step") is not None and row.get("phase") == "after_step_compute"
    ]
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if losses:
        (metrics_dir / "loss.svg").write_text(polyline_svg(losses, "Training loss", "step", "loss"), encoding="utf-8")
    if memory_points:
        (metrics_dir / "peak_allocated_memory.svg").write_text(
            polyline_svg(memory_points, "Peak allocated CUDA memory", "step", "MiB"), encoding="utf-8"
        )
    summary = {
        "elapsed_seconds_launcher": args.elapsed_seconds,
        "loss": {"logged_steps": len(losses), "first": losses[0][1] if losses else None, "last": losses[-1][1] if losses else None, "minimum": min((v for _, v in losses), default=None)},
        "timing": {"profiled_steps": timing.get("step_count"), "mean_step_total_seconds": timing.get("mean_step_total_seconds")},
        "memory": {"peak_allocated_mib_all_ranks": max_allocated, "peak_reserved_mib_all_ranks": max_reserved},
        "artifacts": {"tensorboard": "tensorboard_log/", "loss_csv": "loss.csv", "profiling": "profiling/", "charts": "metrics/*.svg"},
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
