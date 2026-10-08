"""Distributed wall-clock and CUDA-memory profiling for SCoPE smoke runs.

The profiler deliberately measures only training setup and training steps.  It
does not include checkpoint serialization, whose FSDP state-dict gathering can
otherwise dominate a one-sample smoke test.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


_MIB = 1024**2


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_memory(device: torch.device) -> dict[str, float]:
    """Return allocator statistics for the current CUDA device in MiB."""
    if device.type != "cuda":
        return {}
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / _MIB, 1),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / _MIB, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / _MIB, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / _MIB, 1),
    }


@dataclass
class TrainingProfiler:
    """Collect per-rank memory samples and max-rank wall-clock timings."""

    accelerator: Any
    output_path: str | Path
    enabled: bool = True
    profile_steps: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    memory_records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.device = self.accelerator.device

    def should_profile_step(self, step: int) -> bool:
        return self.enabled and (self.profile_steps <= 0 or step <= self.profile_steps)

    def start_timer(self) -> float:
        _synchronize(self.device)
        return time.perf_counter()

    def elapsed_max_rank(self, started_at: float) -> float:
        _synchronize(self.device)
        elapsed = time.perf_counter() - started_at
        if self.accelerator.num_processes == 1:
            return elapsed
        value = torch.tensor(elapsed, device=self.device, dtype=torch.float64)
        return float(self.accelerator.reduce(value, reduction="max").item())

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            _synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory_snapshot(self, *, step: int | None, phase: str) -> None:
        if not self.enabled or self.device.type != "cuda":
            return
        self.memory_records.append({
            "rank": self.accelerator.process_index,
            "step": step,
            "phase": phase,
            **_cuda_memory(self.device),
        })

    def timing_record(self, *, step: int | None, phase: str, seconds_max_rank: float) -> None:
        if self.enabled and self.accelerator.is_main_process:
            self.records.append({"step": step, "phase": phase, "seconds_max_rank": seconds_max_rank})

    def write(self) -> None:
        """Write rank-local memory JSONL and rank-0 timing summary."""
        if not self.enabled:
            return
        profile_dir = self.output_path / "profiling"
        profile_dir.mkdir(parents=True, exist_ok=True)
        memory_path = profile_dir / f"memory_rank{self.accelerator.process_index}.jsonl"
        with memory_path.open("w", encoding="utf-8") as file:
            for record in self.memory_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        step_totals = [record["seconds_max_rank"] for record in self.records if record["phase"] == "step_total"]
        summary = {
            "timing_scope": "model construction, FSDP preparation, then data/forward/backward/optimizer; checkpoint export excluded",
            "rank_aggregation": "maximum wall time across all ranks",
            "memory_scope": "rank-local allocator snapshots; use the largest peak_*_mib across memory_rank*.jsonl",
            "step_count": len(step_totals),
            "step_total_seconds": step_totals,
            "mean_step_total_seconds": sum(step_totals) / len(step_totals) if step_totals else None,
            "records": self.records,
        }
        (profile_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
