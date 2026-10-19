"""Train target-video-only SCoPE adapters on MiniMax H3 Ref2VA.

Run this from a machine where DiffSynth-Studio and its training dependencies
are installed. The script is independent of the upstream source tree: it only
imports it and attaches modules at runtime.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adapter import install_scope_h3, set_scope_camera
from .profiling import TrainingProfiler

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, ToAbsolutePath
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTMiniMaxH3AudioVideoLoss,
    ModelLogger,
    add_general_config,
    add_video_size_config,
)
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.minimax_h3 import MiniMaxH3ReferenceLoader


FRAME_RATE, TIME_FACTOR, TIME_REMAINDER = 24, 17, 5


def wrap_dit_and_text_encoder_with_fsdp(model: "ScopeH3TrainingModule", accelerator: Any) -> None:
    """Shard only DiT and the text encoder; keep both VAEs on CPU.

    This is deliberately *manual* FSDP.  Accelerate must not wrap the outer
    training module: doing so flattens the frozen VAE/text/DiT tree at the
    root and recreates the 120+ GiB per-rank allocation that caused the OOM.
    The layout matches the previously validated ``09_09_fsdp_dit_text`` run.
    """
    if getattr(model, "_scope_manual_fsdp_ready", False):
        return
    if accelerator.num_processes < 2:
        model.pipe.device = accelerator.device
        model.pipe.device_type = accelerator.device.type
        return

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
    from diffsynth.models.minimax_h3_dit import MiniMaxH3DiTBlock

    device = accelerator.device
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        # The DiT root receives camera coordinates and diffusion times along
        # with model activations.  FSDP otherwise defaults to casting every
        # floating-point root input to param_dtype, which quantizes those
        # continuous conditioning values to BF16.  Keep inputs at the dtype
        # produced by the pipeline while FSDP still materializes parameters
        # and reduces gradients in BF16.
        cast_root_forward_inputs=False,
    )
    common = dict(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
    )

    # DiT has a known repeated-block class, so each H3 block becomes an FSDP
    # unit.  This keeps all-gather peaks bounded during the denoising forward.
    dit_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MiniMaxH3DiTBlock},
    )
    model.pipe.dit = FSDP(model.pipe.dit, auto_wrap_policy=dit_policy, **common)

    # The Qwen3-VL text backbone is version-dependent across DiffSynth
    # checkouts.  Size-based wrapping avoids hard-coding a transformers class
    # name while still sharding its decoder layers rather than flattening the
    # entire encoder into one root parameter.
    text_policy = partial(size_based_auto_wrap_policy, min_num_params=8_000_000)
    model.pipe.text_encoder = FSDP(model.pipe.text_encoder, auto_wrap_policy=text_policy, **common)

    # VAEs are frozen and are used only by preprocessing pipeline units.  They
    # are moved to the local GPU only for one such unit, then returned to CPU.
    model.pipe.video_vae.to("cpu")
    model.pipe.audio_vae.to("cpu")
    model.pipe.device = device
    model.pipe.device_type = device.type
    model.pipe.vram_management_enabled = False
    model._scope_manual_fsdp_ready = True


def _run_pipeline_units_with_cpu_vae(model: "ScopeH3TrainingModule", inputs):
    """Run H3 units while transiently onloading frozen VAEs to the local GPU."""
    inputs_shared, inputs_posi, inputs_nega = inputs
    pipe = model.pipe
    for unit in pipe.units:
        onload_names = set(getattr(unit, "onload_model_names", ()) or ())
        vae_names = onload_names.intersection({"video_vae", "audio_vae"})
        if vae_names:
            for name in vae_names:
                getattr(pipe, name).to(pipe.device)
        try:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )
        finally:
            if vae_names:
                for name in vae_names:
                    getattr(pipe, name).to("cpu")
                if pipe.device.type == "cuda":
                    torch.cuda.empty_cache()
    return inputs_shared, inputs_posi, inputs_nega


def save_scope_adapter(accelerator: Any, model: "ScopeH3TrainingModule", output_path: str, file_name: str) -> None:
    """Export only SCoPE weights without all-gathering the full DiT.

    Each H3 block is gathered separately.  Calling ``state_dict()`` on the
    outer DiT FSDP module would all-gather the whole frozen DiT just to save
    ~100 MiB of adapters, defeating the memory layout used for training.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from safetensors.torch import save_file

    accelerator.wait_for_everyone()
    dit = model.pipe.dit
    dit_module = dit.module if isinstance(dit, FSDP) else dit
    state_dict: dict[str, torch.Tensor] = {}
    for block_id, block in enumerate(dit_module.blocks):
        if isinstance(block, FSDP):
            context = FSDP.summon_full_params(block, recurse=False, writeback=False)
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            block_module = block.module if isinstance(block, FSDP) else block
            if accelerator.is_main_process:
                for name, parameter in block_module.named_parameters():
                    if ".scope_pe." in name:
                        state_dict[f"blocks.{block_id}.{name}"] = parameter.detach().cpu().contiguous()

    if accelerator.is_main_process:
        if not state_dict:
            raise RuntimeError("manual FSDP export found no SCoPE parameters")
        destination = Path(output_path)
        destination.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(destination / file_name))
        print(f"Saved {len(state_dict)} SCoPE tensors to {destination / file_name}")
    accelerator.wait_for_everyone()


def load_scope_adapter(model: "ScopeH3TrainingModule", adapter_path: str) -> None:
    """Load a previously exported adapter before manual FSDP wrapping.

    Optimizer moments are intentionally not restored: this is a lightweight,
    interruption-tolerant SFT resume path that preserves all learned SCoPE
    weights while recreating AdamW state locally.
    """
    from safetensors.torch import load_file

    path = Path(adapter_path)
    if not path.is_file():
        raise FileNotFoundError(f"resume adapter does not exist: {path}")
    state_dict = load_file(str(path), device="cpu")
    expected = {name for name, parameter in model.pipe.dit.named_parameters() if ".scope_pe." in name and parameter.requires_grad}
    provided = set(state_dict)
    missing = expected - provided
    unexpected = provided - expected
    if missing or unexpected:
        raise ValueError(
            f"{path}: adapter keys do not match this model; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.pipe.dit.load_state_dict(state_dict, strict=False)
    print(f"Resumed {len(state_dict)} SCoPE tensors from {path}")


class LoadCameraPose:
    """Load one metadata-relative `.npy` pose array as float32 `[F,3|4,4]`."""

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)

    def __call__(self, value: str | list[Any]) -> torch.Tensor:
        path = Path(value)
        path = path if path.is_absolute() else self.base_path / path
        poses = np.load(path, allow_pickle=False)
        if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
            raise ValueError(f"{path}: expected poses [F,3,4] or [F,4,4], got {poses.shape}")
        if not np.isfinite(poses).all():
            raise ValueError(f"{path}: poses contain non-finite values")
        return torch.from_numpy(poses.astype(np.float32, copy=False))


class ScopeH3TrainingModule(DiffusionTrainingModule):
    """DiffSynth H3 SFT module that freezes H3 and trains only `scope_pe`."""

    def __init__(
        self,
        *,
        model_paths: str | None,
        model_id_with_origin_paths: str | None,
        processor_path: str,
        scale_gate_hidden: int,
        learning_rate: float,
        weight_decay: float,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
        training_cfg_scale: float,
        audio_loss_weight: float,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        if training_cfg_scale < 1:
            raise ValueError("training_cfg_scale must be >= 1")
        configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, device=device)
        self.pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=configs,
            processor_config=self.parse_path_or_model_id(processor_path),
        )
        install_scope_h3(self.pipe, scale_gate_hidden=scale_gate_hidden)
        for parameter in self.pipe.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.pipe.dit.named_parameters():
            if ".scope_pe." in name:
                parameter.requires_grad_(True)

        trainable = [name for name, p in self.pipe.dit.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("no SCoPE parameters were marked trainable")
        print(f"Training {len(trainable)} SCoPE tensors only; all original H3 weights are frozen.")
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.scheduler_audio.set_timesteps(1000, training=True)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.training_cfg_scale = training_cfg_scale
        self.audio_loss_weight = audio_loss_weight

    def get_pipeline_inputs(self, data: dict[str, Any]):
        if "camera_poses" not in data or "camera_x_fov" not in data:
            raise KeyError("every sample requires camera_poses and camera_x_fov")
        return {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "input_audio": data.get("input_audio"),
            "references": data.get("references"),
            "ref_image_short_edge": 2048,
            "ref_video_short_edge": 768,
            "ref_video_max_pixels": 768 * 1344,
            "imgvid_cond_noise_aug": self.pipe.imgvid_cond_noise_aug,
            "audio_cond_noise_aug": self.pipe.audio_cond_noise_aug,
            "camera_poses": data["camera_poses"],
            "camera_x_fov": float(data["camera_x_fov"]),
            "tiled": True,
            "tile_size": 256,
            "tile_overlap": 64,
            "cfg_scale": self.training_cfg_scale,
            "seed": 42,
            "rand_device": "cpu",
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }, {"prompt": data["prompt"]}, {"negative_prompt": " "}

    def forward(self, data: dict[str, Any], inputs=None):
        inputs = self.get_pipeline_inputs(data) if inputs is None else inputs
        inputs_shared, inputs_posi, inputs_nega = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        # The camera is held by the runtime adapter; stock pipeline units leave
        # these two metadata fields untouched and the wrapped model fn consumes it.
        set_scope_camera(self.pipe, inputs_shared["camera_poses"], inputs_shared["camera_x_fov"])
        inputs_shared, inputs_posi, inputs_nega = _run_pipeline_units_with_cpu_vae(
            self, (inputs_shared, inputs_posi, inputs_nega)
        )
        return FlowMatchSFTMiniMaxH3AudioVideoLoss(
            self.pipe,
            training_cfg_scale=self.training_cfg_scale,
            audio_loss_weight=self.audio_loss_weight,
            inputs_nega=inputs_nega,
            **inputs_shared,
            **inputs_posi,
        )


def parser() -> argparse.ArgumentParser:
    result = add_video_size_config(add_general_config(argparse.ArgumentParser(description=__doc__)))
    result.add_argument("--processor_path", required=True, help="DiffSynth ModelConfig form, e.g. MiniMaxAI/MiniMax-H3:Ref2VA/processor/")
    result.add_argument("--scale_gate_hidden", type=int, default=64)
    result.add_argument("--audio_loss_weight", type=float, default=0.0, help="Start at 0: retain audio tokens but optimize camera/video only.")
    result.add_argument("--training_cfg_scale", type=float, default=1.0)
    result.add_argument("--enable_profiling", action="store_true", help="Record max-rank step timing and CUDA allocator peaks under output_path/profiling.")
    result.add_argument("--profile_steps", type=int, default=0, help="Profile at most this many optimizer steps; 0 profiles all steps.")
    result.add_argument("--max_steps", type=int, default=None, help="Stop after this total optimizer-step number.")
    result.add_argument("--initial_step", type=int, default=0, help="Completed step number represented by --resume_adapter_path.")
    result.add_argument("--resume_adapter_path", default=None, help="Adapter-only safetensors checkpoint to load before training.")
    result.add_argument("--eval_only", action="store_true", help="Run distributed validation loss only; do not update parameters.")
    result.add_argument("--eval_output_path", default=None, help="Write validation summary JSON here (defaults under output_path).")
    return result


def launch_profiled_training_task(accelerator, dataset, model, model_logger, args, profiler: TrainingProfiler) -> None:
    """DiffSynth's SFT loop with explicit, synchronized profiling boundaries."""
    from diffsynth.diffusion.runner import (
        get_optimizer_class,
        initialize_deepspeed_gradient_checkpointing,
        save_training_args,
    )

    if accelerator.is_main_process:
        save_training_args(args)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda items: items[0], num_workers=args.dataset_num_workers)

    setup_started = profiler.start_timer()
    # FSDP flattens parameters per wrapped H3 block and therefore requires a
    # uniform parameter dtype.  The released H3 checkpoint is BF16, whereas
    # newly constructed SCoPE gates default to FP32.  Cast only the adapter
    # parameters before FSDP wrapping; camera geometry is still computed in
    # FP32 in the adapter forward path.
    for name, parameter in model.named_parameters():
        if ".scope_pe." in name and parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)

    # Do not call ``accelerator.prepare(model, ...)`` here.  Its FSDP wrapper
    # owns the root module and therefore flattens frozen VAE/text/DiT weights
    # together.  Manual FSDP below shards only text encoder + DiT; the frozen
    # VAEs stay on CPU and are transiently onloaded by individual units.
    wrap_dit_and_text_encoder_with_fsdp(model, accelerator)
    optimizer = get_optimizer_class(args.customized_optimizer)(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    profiler.timing_record(step=None, phase="manual_fsdp_dit_text_prepare", seconds_max_rank=profiler.elapsed_max_rank(setup_started))
    profiler.memory_snapshot(step=None, phase="after_manual_fsdp_dit_text_prepare")

    completed_steps = args.initial_step
    ran_steps = False
    for _epoch_id in range(args.num_epochs):
        data_iter = iter(dataloader)
        while True:
            if args.max_steps is not None and completed_steps >= args.max_steps:
                break
            fetch_started = profiler.start_timer()
            try:
                data = next(data_iter)
            except StopIteration:
                break
            completed_steps += 1
            ran_steps = True
            profile_this_step = profiler.should_profile_step(completed_steps)
            if profile_this_step:
                profiler.reset_peak_memory()
                profiler.memory_snapshot(step=completed_steps, phase="after_data_fetch")
                profiler.timing_record(step=completed_steps, phase="data_fetch", seconds_max_rank=profiler.elapsed_max_rank(fetch_started))
                step_started = profiler.start_timer()

            with accelerator.accumulate(model):
                forward_started = profiler.start_timer() if profile_this_step else 0.0
                loss = model(data)
                if profile_this_step:
                    profiler.timing_record(step=completed_steps, phase="forward", seconds_max_rank=profiler.elapsed_max_rank(forward_started))
                    profiler.memory_snapshot(step=completed_steps, phase="after_forward")
                backward_started = profiler.start_timer() if profile_this_step else 0.0
                accelerator.backward(loss)
                if profile_this_step:
                    profiler.timing_record(step=completed_steps, phase="backward", seconds_max_rank=profiler.elapsed_max_rank(backward_started))
                    profiler.memory_snapshot(step=completed_steps, phase="after_backward")
                optimizer_started = profiler.start_timer() if profile_this_step else 0.0
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if profile_this_step:
                    profiler.timing_record(step=completed_steps, phase="optimizer_step", seconds_max_rank=profiler.elapsed_max_rank(optimizer_started))
                    profiler.memory_snapshot(step=completed_steps, phase="after_optimizer_step")

            if profile_this_step:
                profiler.timing_record(step=completed_steps, phase="step_total", seconds_max_rank=profiler.elapsed_max_rank(step_started))
                profiler.memory_snapshot(step=completed_steps, phase="after_step_compute")
            # Saving can gather FSDP state and is intentionally outside the
            # profiled step boundary.
            # ModelLogger's generic state_dict path would all-gather the full
            # nested FSDP DiT.  It remains responsible for scalar logs; the
            # adapter-only exporter below gathers one block at a time.
            model_logger.on_step_end(accelerator, model, save_steps=None, loss=loss)
            if args.save_steps is not None and completed_steps % args.save_steps == 0:
                save_scope_adapter(accelerator, model, args.output_path, f"step-{completed_steps}.safetensors")
        if args.max_steps is not None and completed_steps >= args.max_steps:
            break

    # Checkpointing is intentionally not included in timing/memory peaks.
    if ran_steps and (args.save_steps is None or completed_steps % args.save_steps != 0):
        save_scope_adapter(accelerator, model, args.output_path, f"step-{completed_steps}.safetensors")
    model_logger.on_training_end(accelerator, model, save_steps=None)
    profiler.write()


def launch_validation_task(accelerator, dataset, model, args) -> None:
    """Evaluate each validation record exactly once across manual-FSDP ranks."""
    for name, parameter in model.named_parameters():
        if ".scope_pe." in name and parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    wrap_dit_and_text_encoder_with_fsdp(model, accelerator)
    model.eval()
    local_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    local_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    # FSDP forward calls are collective.  Pad the final global batch so every
    # rank executes the same number of forwards, but exclude padded samples
    # from the reported sum/count.
    steps_per_rank = (len(dataset) + accelerator.num_processes - 1) // accelerator.num_processes
    with torch.no_grad():
        for local_step in range(steps_per_rank):
            index = accelerator.process_index + local_step * accelerator.num_processes
            is_real_sample = index < len(dataset)
            loss = model(dataset[index if is_real_sample else index % len(dataset)])
            if is_real_sample:
                local_loss_sum += loss.detach().to(device=accelerator.device, dtype=torch.float64)
                local_count += 1
    totals = accelerator.reduce(torch.stack((local_loss_sum, local_count)), reduction="sum")
    if accelerator.is_main_process:
        summary = {
            "records": int(totals[1].item()),
            "mean_loss": float((totals[0] / totals[1]).item()),
            "checkpoint": args.resume_adapter_path,
        }
        path = Path(args.eval_output_path or Path(args.output_path) / "validation_loss.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
    accelerator.wait_for_everyone()


def main() -> None:
    args = parser().parse_args()
    if args.task != "sft":
        raise ValueError("this first-run script intentionally supports direct --task sft only; do not use cache splitting before pose caching is implemented")
    if args.num_frames % TIME_FACTOR != TIME_REMAINDER:
        raise ValueError(f"num_frames must be 17n+5 (e.g. 39, 56, 124), got {args.num_frames}")
    if args.initial_step < 0 or (args.max_steps is not None and args.max_steps <= args.initial_step):
        raise ValueError("max_steps must be greater than non-negative initial_step")
    import accelerate
    from accelerate.utils import DistributedDataParallelKwargs
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    video_operator = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path, max_pixels=args.max_pixels, height=args.height, width=args.width,
        height_division_factor=32, width_division_factor=32, num_frames=args.num_frames,
        time_division_factor=TIME_FACTOR, time_division_remainder=TIME_REMAINDER,
        frame_rate=FRAME_RATE, fix_frame_rate=True,
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_operator,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(num_frames=args.num_frames, time_division_factor=TIME_FACTOR, time_division_remainder=TIME_REMAINDER, frame_rate=FRAME_RATE, fix_frame_rate=True),
            "references": MiniMaxH3ReferenceLoader(base_path=args.dataset_base_path, height=args.height, width=args.width, max_pixels=args.max_pixels, num_frames=args.num_frames, frame_rate=FRAME_RATE),
            "camera_poses": LoadCameraPose(args.dataset_base_path),
        },
    )
    construction_started = time.perf_counter()
    model = ScopeH3TrainingModule(
        model_paths=args.model_paths, model_id_with_origin_paths=args.model_id_with_origin_paths,
        processor_path=args.processor_path, scale_gate_hidden=args.scale_gate_hidden,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        training_cfg_scale=args.training_cfg_scale, audio_loss_weight=args.audio_loss_weight,
        device=accelerator.device,
    )
    if args.resume_adapter_path is not None:
        load_scope_adapter(model, args.resume_adapter_path)
    if args.eval_only:
        launch_validation_task(accelerator, dataset, model, args)
        return
    profiler = TrainingProfiler(accelerator, args.output_path, enabled=args.enable_profiling, profile_steps=args.profile_steps)
    if args.enable_profiling:
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        construction_seconds = time.perf_counter() - construction_started
        if accelerator.num_processes > 1:
            construction_seconds = float(accelerator.reduce(torch.tensor(construction_seconds, device=accelerator.device, dtype=torch.float64), reduction="max").item())
        profiler.timing_record(step=None, phase="model_weight_construction", seconds_max_rank=construction_seconds)
    logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt="pipe.dit.",
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_csv_log=args.enable_csv_log,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        initial_step=args.initial_step,
    )
    launch_profiled_training_task(accelerator, dataset, model, logger, args=args, profiler=profiler)


def exclude_quantized_params_from_ddp_sync(*args, **kwargs):
    """No-op compatibility shim for older, non-quantized DiffSynth setups."""


if __name__ == "__main__":
    main()
