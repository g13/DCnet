# Shared Loader And DDP Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `train.py` run on the shared qCLEVR dataset and support true 2-GPU training through PyTorch DDP.

**Architecture:** Use `data2_shared_cues.py` as the active loader, keep single-process behavior as the default, and enable distributed mode only when `WORLD_SIZE > 1`. Add small helpers for batch unpacking, distributed context, metric reduction, and rank-0 side effects instead of introducing a larger training framework.

**Tech Stack:** Python 3.12, PyTorch, torch.distributed, DistributedDataParallel, Hydra/OmegaConf, pytest, Pixi.

---

## Execution Notes

- Do not touch unrelated untracked generated flowchart files.
- Do not commit unless the user explicitly requests commits in the implementation session. Commit steps below are approval-only checkpoints.
- Preserve existing single-process training and the current model-structure-only smoke path.
- Prefer small local helpers in `train.py`; do not create a new training package.

### Task 1: Accept Shared-Loader Batches In Training Loops

**Files:**
- Modify: `train.py`
- Modify: `tests/test_training_options.py`

**Step 1: Write failing tests for 4-item batches**

Add tests that prove `train_iter` and `eval_iter` accept batches shaped like the shared loader output.

```python
def make_shared_batches(num_batches):
    batches = []
    for i in range(num_batches):
        cue = torch.zeros(1, 1)
        mixture = torch.tensor([[float(i + 1)]])
        label = torch.tensor([i % 2], dtype=torch.long)
        batches.append((cue, mixture, label, ["color"]))
    return batches

def test_train_iter_accepts_shared_loader_four_tuple_batches(self):
    model = TinyCueSceneModel()
    optimizer = CountingSGD(model.parameters())

    train_iter(
        make_config(accumulation_steps=1),
        model,
        optimizer,
        None,
        nn.CrossEntropyLoss(),
        make_shared_batches(1),
        lambda _: None,
        epoch=0,
        device=torch.device("cpu"),
    )

    self.assertEqual(optimizer.step_count, 1)

def test_eval_iter_accepts_shared_loader_four_tuple_batches(self):
    model = TinyCueSceneModel()

    loss, acc = eval_iter(
        make_config(),
        model,
        nn.CrossEntropyLoss(),
        make_shared_batches(1),
        lambda _: None,
        epoch=0,
        device=torch.device("cpu"),
    )

    self.assertIsInstance(loss, float)
    self.assertGreaterEqual(acc, 0.0)
```

**Step 2: Run the focused tests and verify failure**

Run: `pixi run python -m pytest tests/test_training_options.py -q`

Expected: FAIL because `train_iter` and `eval_iter` currently unpack only 3 values.

**Step 3: Add a batch-unpack helper**

Add this helper near `_optimizer_steps_per_epoch` in `train.py`.

```python
def _unpack_batch(batch):
    if len(batch) == 3:
        cue, mixture, labels = batch
        return cue, mixture, labels
    if len(batch) >= 4:
        cue, mixture, labels, *_ = batch
        return cue, mixture, labels
    raise ValueError(f"Expected batch with at least 3 items, got {len(batch)}")
```

Update loop headers:

```python
for i, batch in enumerate(bar):
    cue, mixture, labels = _unpack_batch(batch)
```

and:

```python
for batch in val_loader:
    cue, mixture, labels = _unpack_batch(batch)
```

**Step 4: Run focused tests and verify pass**

Run: `pixi run python -m pytest tests/test_training_options.py -q`

Expected: PASS.

**Step 5: Optional commit checkpoint**

Only if the user explicitly requested commits:

Run: `git add train.py tests/test_training_options.py && git commit -m "fix: accept shared qclevr batches"`

### Task 2: Add Data Loader Distributed Sampler Support

**Files:**
- Modify: `data2_shared_cues.py`
- Create: `tests/test_data2_shared_cues.py`

**Step 1: Write failing tests for sampler wiring**

Patch dataset construction so the test does not need real images.

```python
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, SequentialSampler

import data2_shared_cues


class TinyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.items = [0, 1, 2, 3]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class SharedCueLoaderTests(unittest.TestCase):
    @patch("data2_shared_cues.qCLEVRDataset", TinyDataset)
    def test_single_process_train_loader_shuffles(self):
        train_loader, val_loader = data2_shared_cues.get_qclevr_dataloaders(
            data_root="unused",
            assets_path="unused",
            train_batch_size=2,
            val_batch_size=2,
            resolution=(128, 128),
            num_workers=0,
        )

        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)

    @patch("data2_shared_cues.qCLEVRDataset", TinyDataset)
    def test_distributed_loaders_use_distributed_samplers(self):
        train_loader, val_loader = data2_shared_cues.get_qclevr_dataloaders(
            data_root="unused",
            assets_path="unused",
            train_batch_size=2,
            val_batch_size=2,
            resolution=(128, 128),
            num_workers=0,
            distributed=True,
            rank=1,
            world_size=2,
        )

        self.assertIsInstance(train_loader.sampler, DistributedSampler)
        self.assertIsInstance(val_loader.sampler, DistributedSampler)
        self.assertTrue(train_loader.sampler.shuffle)
        self.assertFalse(val_loader.sampler.shuffle)
```

**Step 2: Run tests and verify failure**

Run: `pixi run python -m pytest tests/test_data2_shared_cues.py -q`

Expected: FAIL because `distributed`, `rank`, and `world_size` are not accepted yet.

**Step 3: Extend `get_qclevr_dataloaders`**

Update imports:

```python
from torch.utils.data import DataLoader, Dataset, DistributedSampler
```

Update the function signature:

```python
def get_qclevr_dataloaders(
    data_root: str,
    assets_path: str,
    train_batch_size: int,
    val_batch_size: int,
    resolution: tuple[int, int],
    holdout: list = [],
    mode: str = "color",
    primitive: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
```

After dataset construction, add:

```python
train_sampler = None
val_sampler = None
if distributed:
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed or 0,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
```

Update DataLoader construction:

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    ...
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    sampler=val_sampler,
    ...
)
```

**Step 4: Run tests and verify pass**

Run: `pixi run python -m pytest tests/test_data2_shared_cues.py -q`

Expected: PASS.

**Step 5: Optional commit checkpoint**

Only if the user explicitly requested commits:

Run: `git add data2_shared_cues.py tests/test_data2_shared_cues.py && git commit -m "feat: add distributed qclevr samplers"`

### Task 3: Add Distributed Context And Model Unwrap Helpers

**Files:**
- Modify: `train.py`
- Create: `tests/test_distributed_training.py`

**Step 1: Write failing helper tests**

```python
import os
import unittest
from unittest.mock import patch

import torch.nn as nn

import train


class Wrapper:
    def __init__(self, module):
        self.module = module


class CompiledWrapper:
    def __init__(self, module):
        self._orig_mod = module


class DistributedTrainingHelperTests(unittest.TestCase):
    def test_context_defaults_to_single_process(self):
        with patch.dict(os.environ, {}, clear=True):
            context = train._distributed_context_from_env()

        self.assertFalse(context.enabled)
        self.assertEqual(context.rank, 0)
        self.assertEqual(context.local_rank, 0)
        self.assertEqual(context.world_size, 1)

    def test_context_reads_torchrun_environment(self):
        env = {"LOCAL_RANK": "1", "RANK": "3", "WORLD_SIZE": "4"}
        with patch.dict(os.environ, env, clear=True):
            context = train._distributed_context_from_env()

        self.assertTrue(context.enabled)
        self.assertEqual(context.local_rank, 1)
        self.assertEqual(context.rank, 3)
        self.assertEqual(context.world_size, 4)

    def test_unwrap_model_handles_ddp_and_compile_wrappers(self):
        model = nn.Linear(1, 1)
        wrapped = Wrapper(CompiledWrapper(model))

        self.assertIs(train._unwrap_model(wrapped), model)
```

**Step 2: Run tests and verify failure**

Run: `pixi run python -m pytest tests/test_distributed_training.py -q`

Expected: FAIL because helper functions do not exist.

**Step 3: Add helpers to `train.py`**

Add imports:

```python
from dataclasses import dataclass

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
```

Add helpers near the existing top-level helpers:

```python
@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    local_rank: int
    rank: int
    world_size: int


def _distributed_context_from_env() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return DistributedContext(
        enabled=world_size > 1,
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        rank=int(os.environ.get("RANK", "0")),
        world_size=world_size,
    )


def _is_main_process(context: DistributedContext) -> bool:
    return context.rank == 0


def _unwrap_model(model):
    unwrapped = getattr(model, "module", model)
    return getattr(unwrapped, "_orig_mod", unwrapped)
```

**Step 4: Run tests and verify pass**

Run: `pixi run python -m pytest tests/test_distributed_training.py -q`

Expected: PASS.

**Step 5: Optional commit checkpoint**

Only if the user explicitly requested commits:

Run: `git add train.py tests/test_distributed_training.py && git commit -m "feat: add distributed training helpers"`

### Task 4: Reduce Metrics Across Ranks

**Files:**
- Modify: `train.py`
- Modify: `tests/test_distributed_training.py`
- Modify: `tests/test_training_options.py`

**Step 1: Write tests for non-distributed metric reduction**

Add a lightweight test that does not initialize a process group.

```python
def test_reduce_metrics_noops_when_not_distributed(self):
    context = train.DistributedContext(enabled=False, local_rank=0, rank=0, world_size=1)

    loss_sum, correct, total = train._reduce_metrics(
        loss_sum=3.5,
        correct=2,
        total=4,
        device=torch.device("cpu"),
        context=context,
    )

    self.assertEqual(loss_sum, 3.5)
    self.assertEqual(correct, 2)
    self.assertEqual(total, 4)
```

Also update existing `train_iter` and `eval_iter` calls in tests to pass a default context only if the implementation requires it. Prefer keeping the public function signatures backward-compatible by defaulting `context=None`.

**Step 2: Run tests and verify failure**

Run: `pixi run python -m pytest tests/test_distributed_training.py tests/test_training_options.py -q`

Expected: FAIL because `_reduce_metrics` does not exist.

**Step 3: Implement metric reduction helper**

```python
def _reduce_metrics(loss_sum, correct, total, device, context: DistributedContext):
    if not context.enabled:
        return loss_sum, correct, total
    metrics = torch.tensor(
        [float(loss_sum), float(correct), float(total)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[0].item(), int(metrics[1].item()), int(metrics[2].item())
```

Update `train_iter` and `eval_iter` to accumulate weighted loss sums:

```python
loss_sum += loss.item() * labels.size(0)
correct += (predicted == labels).sum().item()
total += labels.size(0)
```

At the end, reduce sums and compute:

```python
loss = loss_sum / total
acc = correct / total
```

Keep `context` optional:

```python
context = context or DistributedContext(False, 0, 0, 1)
```

**Step 4: Run tests and verify pass**

Run: `pixi run python -m pytest tests/test_distributed_training.py tests/test_training_options.py -q`

Expected: PASS.

**Step 5: Optional commit checkpoint**

Only if the user explicitly requested commits:

Run: `git add train.py tests/test_distributed_training.py tests/test_training_options.py && git commit -m "feat: reduce distributed metrics"`

### Task 5: Wire DDP Into `train.py`

**Files:**
- Modify: `train.py`

**Step 1: Switch the active loader import**

Change:

```python
from data import get_qclevr_dataloaders
```

to:

```python
from data2_shared_cues import get_qclevr_dataloaders
```

**Step 2: Initialize distributed context and device**

At the start of `train`, after converting config:

```python
context = _distributed_context_from_env()
if context.enabled:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
if torch.cuda.is_available():
    if context.enabled:
        torch.cuda.set_device(context.local_rank)
        device = torch.device("cuda", context.local_rank)
    else:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

Wrap the body in `try/finally` or ensure cleanup near the end:

```python
if context.enabled and dist.is_initialized():
    dist.destroy_process_group()
```

**Step 3: Gate rank-0 side effects**

Only run these on `_is_main_process(context)`:

- model setup report print
- Mermaid diagram export
- dataset debug prints
- wandb init and wandb log function
- checkpoint directory creation
- checkpoint saves and checkpoint copy
- training plots
- final history save

For non-main ranks, use:

```python
wandb_log = lambda x: None
```

**Step 4: Compile, then wrap with DDP**

After `torch.compile`, add:

```python
if context.enabled:
    ddp_kwargs = {}
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [context.local_rank]
        ddp_kwargs["output_device"] = context.local_rank
    model = DistributedDataParallel(model, **ddp_kwargs)
```

Then create the optimizer from `model.parameters()`.

**Step 5: Pass distributed settings to loaders**

```python
train_loader, val_loader = get_qclevr_dataloaders(
    ...,
    distributed=context.enabled,
    rank=context.rank,
    world_size=context.world_size,
)
```

Before each epoch:

```python
if hasattr(train_loader.sampler, "set_epoch"):
    train_loader.sampler.set_epoch(epoch)
```

**Step 6: Pass context into train/eval iterations**

```python
train_loss, train_acc = train_iter(..., device, context=context)
test_loss, test_acc = eval_iter(..., device, context=context)
```

**Step 7: Save unwrapped model state**

Change checkpoint save to:

```python
"model_state_dict": _unwrap_model(model).state_dict(),
```

**Step 8: Run syntax and focused tests**

Run: `pixi run python -m pytest tests/test_training_options.py tests/test_distributed_training.py tests/test_data2_shared_cues.py -q`

Expected: PASS.

Run: `pixi run python -m py_compile train.py data2_shared_cues.py`

Expected: no output.

**Step 9: Optional commit checkpoint**

Only if the user explicitly requested commits:

Run: `git add train.py && git commit -m "feat: enable ddp training"`

### Task 6: Fix Dataset Debug Summaries For Shared Batches

**Files:**
- Modify: `train.py`

**Step 1: Update label scans to use dataset metadata when available**

Avoid iterating all images through `__getitem__` just to count labels. Replace:

```python
train_labels = [label for _, _, label in train_loader.dataset]
```

with:

```python
train_labels = getattr(train_loader.dataset, "counts", None)
if train_labels is not None:
    print("原始训练集类别分布:", torch.bincount(torch.tensor(train_labels)))
```

Do the same for validation. Keep these prints rank-0-only.

**Step 2: Update first-batch debug unpacking**

```python
for batch_idx, batch in enumerate(train_loader):
    if batch_idx == 0:
        _, _, labels = _unpack_batch(batch)
        print("采样后的首个batch类别分布:", torch.bincount(labels))
        break
```

**Step 3: Run model-structure smoke test**

Run: `pixi run python train.py output_model_structure_only=True`

Expected: prints the model report and skips data loading/training.

**Step 4: Optional commit checkpoint**

Only if the user explicitly requested commits:

Run: `git add train.py && git commit -m "fix: summarize shared qclevr batches"`

### Task 7: Verify Shared Dataset Loading

**Files:**
- No source changes expected unless this smoke test exposes a defect.

**Step 1: Run shared-loader smoke test**

Run:

```bash
pixi run python - <<'PY'
from data2_shared_cues import get_qclevr_dataloaders

train_loader, val_loader = get_qclevr_dataloaders(
    data_root='/scratch/wd/DCnet/data/qclevr_shared/',
    assets_path='data/CLEVR_v1.0/cues',
    train_batch_size=2,
    val_batch_size=2,
    resolution=(128, 128),
    holdout=[],
    mode='every',
    primitive=True,
    num_workers=0,
    seed=42,
)

batch = next(iter(train_loader))
print(len(train_loader.dataset), len(val_loader.dataset))
print(len(batch), tuple(batch[0].shape), tuple(batch[1].shape), tuple(batch[2].shape), batch[3][:2])
PY
```

Expected:

```text
384000 38400
4 (2, 3, 128, 128) (2, 3, 128, 128) (2,) ...
```

**Step 2: Run full test suite**

Run: `pixi run python -m pytest`

Expected: all tests pass.

**Step 3: Check whitespace**

Run: `git diff --check`

Expected: no output.

### Task 8: Prepare 2-GPU Launch Command

**Files:**
- Modify: `docs/reproduction-paper-alignment.md` only if you want to record the launch command.

**Step 1: Decide global batch semantics**

With DDP, global effective batch size is:

```text
data.batch_size * train.accumulation_steps * WORLD_SIZE
```

For paper-like global effective batch 256 on 2 GPUs, use:

```text
data.batch_size=64 train.accumulation_steps=2
```

For the current config unchanged, 2 GPUs gives global effective batch 512:

```text
data.batch_size=128 train.accumulation_steps=2
```

**Step 2: Provide recommended command**

Recommended paper-like 2-GPU command:

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run torchrun --standalone --nproc_per_node=2 train.py data.batch_size=64 data.val_batch_size=64 train.accumulation_steps=2 data.root=/scratch/wd/DCnet/data/qclevr_shared/ data.mode=every
```

If the user wants to prioritize throughput over exact global batch size:

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run torchrun --standalone --nproc_per_node=2 train.py data.root=/scratch/wd/DCnet/data/qclevr_shared/ data.mode=every
```

**Step 3: Optional short DDP smoke test**

Only run this if GPUs are available and a short distributed job is safe. Prefer an override that avoids long training. If no short-run override exists, do not claim runtime DDP verification.

Candidate command:

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run torchrun --standalone --nproc_per_node=2 train.py train.epochs=1 data.batch_size=2 data.val_batch_size=2 train.accumulation_steps=1 data.num_workers=0 checkpoint.disable=True wandb=False data.root=/scratch/wd/DCnet/data/qclevr_shared/ data.mode=every
```

Expected: both ranks initialize, load data, complete one epoch, and rank 0 prints final metrics. Stop and debug if either rank hangs or crashes.

### Task 9: Final Verification And Handoff

**Files:**
- Review all modified files.

**Step 1: Run final verification**

Run: `pixi run python -m pytest`

Expected: PASS.

Run: `pixi run python train.py output_model_structure_only=True`

Expected: model report prints the expected parameter count and skips training.

Run: `pixi run python -m py_compile train.py data2_shared_cues.py`

Expected: no output.

Run: `git diff --check`

Expected: no output.

**Step 2: Inspect diff**

Run: `git diff -- train.py data2_shared_cues.py tests/test_training_options.py tests/test_data2_shared_cues.py tests/test_distributed_training.py docs/reproduction-paper-alignment.md`

Expected: only intentional shared-loader, DDP, test, and optional docs changes.

**Step 3: Report exact status**

Summarize:

- files changed
- tests run and results
- whether a real `torchrun` smoke test was run
- recommended 2-GPU command
- whether any commit was skipped because explicit commit approval was absent

**Step 4: Optional final commit**

Only if the user explicitly requested commits:

Run: `git status --short && git diff && git log --oneline -10`

Then stage only intended files and commit with:

```bash
git add train.py data2_shared_cues.py tests/test_training_options.py tests/test_data2_shared_cues.py tests/test_distributed_training.py docs/plans/2026-05-21-shared-loader-ddp-training-design.md docs/plans/2026-05-21-shared-loader-ddp-training.md
git commit -m "feat: support shared qclevr ddp training"
```
