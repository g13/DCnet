# Shared Loader And DDP Training Design

## Goal

Make the active training path work with the shared qCLEVR dataset at `/scratch/wd/DCnet/data/qclevr_shared/` and support true 2-GPU training with PyTorch DistributedDataParallel.

## Scope

- Use the shared-qCLEVR loader in `data2_shared_cues.py` for `mode: every` mixed color, shape, and conjunction training.
- Keep the existing paper-aligned training settings: effective batch size 256, AMP enabled on CUDA, AdamW at `4e-4`, one-cycle schedule, and 100 epochs.
- Add DDP support for launches through `torchrun`.
- Keep single-process CPU/CUDA training working for smoke tests and smaller runs.
- Do not implement Figure 4 lesions or novel-cue holdout experiments in this task.

## Current State

- `train.py` imports `get_qclevr_dataloaders` from `data.py`, whose old layout assertions fail on the shared dataset.
- `data2_shared_cues.py` can load the shared dataset and supports `mode: every`, but returns 4-tuples: `(cue, image, label, mode)`.
- `train_iter`, `eval_iter`, and debug label scans currently assume 3-tuples: `(cue, image, label)`.
- There is no current DDP, `torchrun`, rank, local-rank, world-size, distributed sampler, or cross-rank metric reduction support.

## Recommended Approach

Use DDP via `torchrun`, not `DataParallel`.

DDP is the better fit because it runs one process per GPU, uses a distributed sampler to split data correctly, avoids the single-process scatter/gather bottleneck, and is the standard PyTorch path for multi-GPU training. `DataParallel` is easier to wire, but it is slower and leaves data sampling, metric handling, and large-model behavior less robust.

## Training Architecture

Add small DDP utilities in `train.py`:

- detect distributed mode from `WORLD_SIZE > 1`
- read `LOCAL_RANK`, `RANK`, and `WORLD_SIZE`
- initialize `torch.distributed` with NCCL on CUDA or Gloo when CUDA is unavailable
- set the CUDA device to `LOCAL_RANK`
- expose `is_main_process` for print/log/checkpoint gates
- destroy the process group at the end

Keep single-process behavior as the default when the script is launched without `torchrun`.

## Data Flow

Switch the active import in `train.py` to the shared loader from `data2_shared_cues.py`.

Update `data2_shared_cues.get_qclevr_dataloaders()` to accept optional train and validation samplers. When a sampler is provided, disable normal `shuffle` so `DistributedSampler` controls ordering. The train sampler should shuffle by epoch; the validation sampler should not shuffle.

Add a batch-unpack helper in `train.py` that accepts either:

- `(cue, mixture, labels)` from old-style tests or compatible loaders
- `(cue, mixture, labels, mode)` from the shared loader

Training and evaluation should ignore `mode` for loss computation, while debug/reporting code can use it for optional mode counts later.

## Model And Optimizer Flow

Create the model on the correct rank-local device, compile it according to config, then wrap it in `DistributedDataParallel` when distributed mode is active.

Optimizer and scheduler should use the wrapped model parameters after compilation/wrapping. Checkpoint state should unwrap DDP and torch-compile wrappers before saving.

The existing gradient accumulation math remains unchanged. Per-process microbatch size remains `config.data.batch_size`; with 2 GPUs, total effective batch becomes `batch_size * accumulation_steps * world_size`. If the paper-faithful global effective batch must remain 256 under DDP, use `batch_size: 64` and `accumulation_steps: 2` for 2 GPUs.

## Metrics And Side Effects

Reduce aggregate loss sums, correct counts, and total counts across ranks before reporting epoch metrics. Avoid averaging already-averaged per-rank accuracies.

Only rank 0 should:

- print model reports and dataset debug summaries
- export Mermaid diagrams
- initialize and log to wandb
- create checkpoint files and symlinks/copies
- save plots and training history

All ranks still participate in training and validation.

## Error Handling

- Fail clearly if `torchrun` requests distributed mode but `torch.distributed` cannot initialize.
- Keep CPU/single-GPU behavior working without requiring distributed environment variables.
- Avoid changing or deleting unrelated untracked generated flowchart files.

## Verification

Run these checks after implementation:

- `pixi run python -m pytest`
- `pixi run python train.py output_model_structure_only=True`
- a shared-loader smoke test against `/scratch/wd/DCnet/data/qclevr_shared/`
- `pixi run python -m py_compile train.py data2_shared_cues.py`
- `git diff --check`

If GPUs are available and a short distributed smoke test is safe, also run a minimal `torchrun --nproc_per_node=2 ...` command with a tiny epoch/step override. If no safe short run exists, report the exact unrun command instead of claiming DDP was runtime-verified.
