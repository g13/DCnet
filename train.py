import os
import shutil
from dataclasses import dataclass
from math import ceil
from typing import Callable, Optional

import hydra
import torch
import torch.distributed as dist
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from data2_shared_cues import get_qclevr_dataloaders
from model import Conv2dEIRNN
from utils import (
    AttrDict,
    export_mermaid_diagram_assets,
    format_mermaid_model_diagram,
    format_model_setup_report,
    get_git_commit_hash,
    seed,
)
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    local_rank: int
    rank: int
    world_size: int


def _distributed_context_from_env() -> DistributedContext:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return DistributedContext(
        enabled=world_size > 1,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
    )


def _is_main_process(context: DistributedContext) -> bool:
    return context.rank == 0


def _device_for_context(context: DistributedContext) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if context.enabled:
        return torch.device("cuda", context.local_rank)
    return torch.device("cuda")


def _setup_distributed(context: DistributedContext) -> torch.device:
    device = _device_for_context(context)
    if context.enabled:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        if device.type == "cuda":
            torch.cuda.set_device(context.local_rank)
    return device


def _wrap_distributed_model(model, device: torch.device, context: DistributedContext):
    if not context.enabled:
        return model
    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    return DistributedDataParallel(
        model,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )


def _cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _reduce_metrics(loss_sum, correct, total, device, context):
    if not context.enabled:
        return loss_sum, correct, total

    values = torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values[0].item(), int(values[1].item()), int(values[2].item())


def _unwrap_model(model):
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
    return model


def _optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    accumulation_steps = max(1, int(accumulation_steps))
    return ceil(num_batches / accumulation_steps)


def _checkpointing_enabled(config: AttrDict) -> bool:
    return not bool(config.checkpoint.get("disable", False))


def _scheduler_disabled(scheduler_fn) -> bool:
    return scheduler_fn is None or str(scheduler_fn).lower() in {"none", "null"}


def _build_scheduler(config: AttrDict, optimizer: torch.optim.Optimizer, num_batches: int):
    if _scheduler_disabled(config.scheduler.fn):
        return None
    if config.scheduler.fn == "one_cycle":
        optimizer_steps_per_epoch = _optimizer_steps_per_epoch(
            num_batches, config.train.accumulation_steps
        )
        return OneCycleLR(
            optimizer,
            max_lr=config.optimizer.lr,
            total_steps=config.train.epochs * optimizer_steps_per_epoch,
            pct_start=config.scheduler.pct_start,
        )
    raise NotImplementedError(f"Scheduler {config.scheduler.fn} not implemented")


def _unpack_batch(batch):
    if len(batch) < 3:
        raise ValueError("Expected batch with at least 3 items")
    return batch[0], batch[1], batch[2]


def _label_counts_from_dataset(dataset):
    counts = getattr(dataset, "counts", None)
    if counts is None:
        return None
    return torch.bincount(torch.as_tensor(counts, dtype=torch.long).reshape(-1))


def _first_batch_label_counts(loader):
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, torch.utils.data.RandomSampler) or getattr(sampler, "shuffle", False):
        return None
    for batch in loader:
        _, _, labels = _unpack_batch(batch)
        return torch.bincount(labels.reshape(-1).to(dtype=torch.long))
    return None


def train_iter(
    config: AttrDict,
    model: Conv2dEIRNN,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    wandb_log: Callable[[dict[str, float, int]], None],
    epoch: int,
    device: torch.device,
    context: Optional[DistributedContext] = None,
) -> tuple[float, float]:
    """
    Perform a single training iteration.

    Args:
        config (AttrDict): Configuration parameters.
        model (Conv2dEIRNN): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        wandb_log (Callable[[dict[str, float, int]], None]): Function to log training statistics to Weights & Biases.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple[float, float]: A tuple containing the training loss and accuracy.
    """
    if context is None:
        context = DistributedContext(False, 0, 0, 1)

    if config.train.grad_clip.disable:
        clip_grad_ = lambda x, y: None
    elif config.train.grad_clip.type == "norm":
        clip_grad_ = lambda x, y: clip_grad_norm_(x, y, foreach=False)
    elif config.train.grad_clip.type == "value":
        clip_grad_ = lambda x, y: clip_grad_value_(x, y, foreach=False)
    else:
        raise NotImplementedError(
            f"Gradient clipping type {config.train.grad_clip_type} not implemented"
        )
    model.train()
    accumulation_steps = max(1, int(config.train.get("accumulation_steps", 1)))
    amp_enabled = bool(config.train.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0
    running_loss_sum = 0.0
    running_correct = 0
    running_total = 0

    bar = tqdm(
        train_loader,
        desc=(f"Training | Epoch: {epoch} | " f"Loss: {0:.4f} | " f"Acc: {0:.2%}"),
        disable=not config.tqdm or not _is_main_process(context),
    )
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(train_loader)
    accumulation_group = []
    for i, batch in enumerate(bar):
        accumulation_group.append((i, batch))
        should_step = (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches
        if not should_step:
            continue

        group_total_samples = sum(
            _unpack_batch(group_batch)[2].size(0)
            for _, group_batch in accumulation_group
        )
        for group_i, group_batch in accumulation_group:
            cue, mixture, labels = _unpack_batch(group_batch)
            # qCLEVR batches are cue image, scene image, and count label:
            # cue: [B, 3, 128, 128], mixture(scene): [B, 3, 128, 128], labels: [B].
            # The labels are class indices for target counts 0..5.
            cue = cue.to(device)
            mixture = mixture.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                # The model internally runs a cue phase and then a scene phase. The default
                # output is a single logits tensor [B, 6]; with `all_timesteps=True` it returns
                # one logits tensor per scene-phase step.
                outputs = model(cue, mixture, all_timesteps=config.criterion.all_timesteps)
                if config.criterion.all_timesteps:
                    losses = []
                    for output in outputs:
                        losses.append(criterion(output, labels))
                    loss = sum(losses) / len(losses)
                    outputs = outputs[-1]
                else:
                    loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            scaler.scale(loss * (batch_size / group_total_samples)).backward()

            # Update statistics
            train_loss_sum += loss.item() * batch_size
            running_loss_sum += loss.item() * batch_size

            predicted = outputs.argmax(-1)
            correct = (predicted == labels).sum().item()
            train_correct += correct
            running_correct += correct
            train_total += batch_size
            running_total += batch_size

            # Log statistics
            if (group_i + 1) % config.train.log_freq == 0:
                running_loss = running_loss_sum / running_total
                running_acc = running_correct / running_total
                wandb_log(dict(running_loss=running_loss, running_acc=running_acc))
                if config.train.get("print_batch_progress", False) and _is_main_process(context):
                    print(
                        f"Training | Epoch: {epoch} | "
                        f"Batch: {group_i + 1}/{num_batches} | "
                        f"Loss: {running_loss:.4f} | "
                        f"Acc: {running_acc:.2%}",
                        flush=True,
                    )
                bar.set_description(
                    f"Training | Epoch: {epoch} | "
                    f"Loss: {running_loss:.4f} | "
                    f"Acc: {running_acc:.2%}"
                )
                running_loss_sum = 0
                running_correct = 0
                running_total = 0

        if not config.train.grad_clip.disable and amp_enabled:
            scaler.unscale_(optimizer)
        clip_grad_(model.parameters(), config.train.grad_clip.value)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accumulation_group = []

    # Calculate average training loss and accuracy
    train_loss_sum, train_correct, train_total = _reduce_metrics(
        train_loss_sum, train_correct, train_total, device, context
    )
    if train_total == 0:
        raise ValueError("Cannot compute training metrics with zero samples")
    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    wandb_log(dict(train_loss=train_loss, train_acc=train_acc))

    return train_loss, train_acc


def eval_iter(
    config: AttrDict,
    model: Conv2dEIRNN,
    criterion: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    wandb_log: Callable[[dict[str, float, int]], None],
    epoch: int,
    device: torch.device,
    context: Optional[DistributedContext] = None,
) -> tuple[float, float]:
    """
    Perform a single evaluation iteration.

    Args:
        model (Conv2dEIRNN): The model to be evaluated.
        criterion (torch.nn.Module): The loss function.
        val_loader (torch.utils.data.DataLoader): The test data loader.
        wandb_log (function): Function to log evaluation statistics to Weights & Biases.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing the test loss and accuracy.
    """
    if context is None:
        context = DistributedContext(False, 0, 0, 1)

    model.eval()
    amp_enabled = bool(config.train.get("amp", False)) and device.type == "cuda"
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in val_loader:
            cue, mixture, labels = _unpack_batch(batch)
            cue = cue.to(device)
            mixture = mixture.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(cue, mixture, all_timesteps=config.criterion.all_timesteps)
                if config.criterion.all_timesteps:
                    losses = []
                    for output in outputs:
                        losses.append(criterion(output, labels))
                    loss = sum(losses) / len(losses)
                    outputs = outputs[-1]
                else:
                    loss = criterion(outputs, labels)

            # Update statistics
            batch_size = labels.size(0)
            test_loss_sum += loss.item() * batch_size
            predicted = outputs.argmax(-1)
            correct = (predicted == labels).sum().item()
            test_correct += correct
            test_total += batch_size

    # Calculate average test loss and accuracy
    test_loss_sum, test_correct, test_total = _reduce_metrics(
        test_loss_sum, test_correct, test_total, device, context
    )
    if test_total == 0:
        raise ValueError("Cannot compute evaluation metrics with zero samples")
    test_loss = test_loss_sum / test_total
    test_acc = test_correct / test_total

    wandb_log(dict(test_loss=test_loss, test_acc=test_acc, epoch=epoch))

    return test_loss, test_acc


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str): Path to save the plot image
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config1.yaml",
)
def train(config: DictConfig) -> None:
    """
    Train the model using the provided configuration.

    Args:
        config (dict): Configuration parameters.
    """
    config = OmegaConf.to_container(config, resolve=True)
    config = AttrDict(config)
    context = _distributed_context_from_env()
    device = _setup_distributed(context)
    try:
        # Set the random seed
        if config.seed is not None:
            seed(config.seed)
        # Set the matmul precision
        torch.set_float32_matmul_precision(config.train.matmul_precision)
        # The main training path always imports `Conv2dEIRNN` from model.py. `model_fig4.py`
        # is an auxiliary Figure 4 reproduction attempt and is not used here by default.
        model = Conv2dEIRNN(**config.model).to(device)
        if _is_main_process(context):
            print(format_model_setup_report(model, include_mermaid=False))
            try:
                diagram_outputs = export_mermaid_diagram_assets(
                    format_mermaid_model_diagram(model),
                    output_dir=config.get("model_diagram_output_dir", "."),
                    commit_hash=get_git_commit_hash(default="unknown"),
                )
                print(
                    "Saved model diagram files to: "
                    f"{diagram_outputs['source_path']}, {diagram_outputs['pdf_path']}"
                )
            except Exception as exc:
                print(f"Warning: could not export model diagram files: {exc}")
        if config.get("output_model_structure_only", False):
            if _is_main_process(context):
                print("Model structure only mode enabled; skipping compile, data loading, and training.")
            return

        # Compile the model if requested
        model = torch.compile(
            model,
            fullgraph=config.compile.fullgraph,
            dynamic=config.compile.dynamic,
            backend=config.compile.backend,
            mode=config.compile.mode,
            disable=config.compile.disable,
        )
        model = _wrap_distributed_model(model, device, context)

        # Initialize the optimizer
        if config.optimizer.fn == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.optimizer.lr,
                momentum=config.optimizer.momentum,
            )
        elif config.optimizer.fn == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.optimizer.lr,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
            )
        elif config.optimizer.fn == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.optimizer.lr,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
            )
        else:
            raise NotImplementedError(f"Optimizer {config.optimizer.fn} not implemented")

        # Initialize the loss function
        if config.criterion.fn == "ce":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Criterion {config.criterion.fn} not implemented")

        # Get the data loaders
        # Dataloaders batch cue images and scene images separately. This code path is the repo's
        # recurrent cue-then-scene implementation, as opposed to an implicit stacked-input baseline.
        train_loader, val_loader = get_qclevr_dataloaders(
            data_root=config.data.root,
            assets_path=config.data.assets_path,
            train_batch_size=config.data.batch_size,
            val_batch_size=config.data.val_batch_size,
            resolution=config.model.input_size,
            holdout=config.data.holdout,
            mode=config.data.mode,
            primitive=config.data.primitive,
            num_workers=config.data.num_workers,
            seed=config.seed,
            distributed=context.enabled,
            rank=context.rank,
            world_size=context.world_size,
        )

        if _is_main_process(context):
            # Debugging statistics added in the current branch. These help inspect class balance,
            # but they are not part of the paper's reported training recipe.
            train_label_counts = _label_counts_from_dataset(train_loader.dataset)
            if train_label_counts is not None:
                print("原始训练集类别分布:", train_label_counts)
            val_label_counts = _label_counts_from_dataset(val_loader.dataset)
            if val_label_counts is not None:
                print("原始验证集类别分布:", val_label_counts)

            # 检查采样后的批次分布
            first_batch_label_counts = _first_batch_label_counts(train_loader)
            if first_batch_label_counts is not None:
                print("采样后的首个batch类别分布:", first_batch_label_counts)

            print(f"训练集总样本数: {len(train_loader.dataset)}")
            print(f"验证集总样本数: {len(val_loader.dataset)}")

        # Initialize the learning rate scheduler
        scheduler = _build_scheduler(config, optimizer, len(train_loader))

        # Initialize Weights & Biases
        if config.wandb and _is_main_process(context):
            wandb.init(project="EI RNN", config=config)
            wandb_log = lambda x: wandb.log(x)
        else:
            wandb_log = lambda x: None

        checkpointing_enabled = _checkpointing_enabled(config)
        if _is_main_process(context) and checkpointing_enabled:
            # Create the checkpoint directory
            if config.wandb:
                checkpoint_dir = os.path.join(config.checkpoint.root, wandb.run.name)
            else:
                checkpoint_dir = config.checkpoint.root
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epochs': []
        }

        for epoch in range(config.train.epochs):
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            # Train the model
            train_loss, train_acc = train_iter(
                config,
                model,
                optimizer,
                scheduler,
                criterion,
                train_loader,
                wandb_log,
                epoch,
                device,
                context=context,
            )

            # Evaluate the model on the validation set
            test_loss, test_acc = eval_iter(
                config, model, criterion, val_loader, wandb_log, epoch, device, context=context
            )

            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['epochs'].append(epoch)

            if _is_main_process(context):
                # Print the epoch statistics
                print(
                    f"Epoch [{epoch}/{config.train.epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Accuracy: {train_acc:.2%} | "
                    f"Test Loss: {test_loss:.4f}, "
                    f"Test Accuracy: {test_acc:.2%}"
                )

            if _is_main_process(context) and checkpointing_enabled:
                # Save the model
                file_path = os.path.abspath(
                    os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
                )
                link_path = os.path.abspath(os.path.join(checkpoint_dir, "checkpoint.pt"))
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": _unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history  # 保存训练历史
                }
                torch.save(checkpoint, file_path)
                try:
                    os.remove(link_path)
                except FileNotFoundError:
                    pass
                shutil.copy2(file_path, link_path)

                # Plot training curves every few epochs or at the end
                if (epoch + 1) % 10 == 0 or epoch == config.train.epochs - 1:
                    plot_save_path = os.path.join(checkpoint_dir, f"training_curves_epoch_{epoch}.png")
                    plot_training_curves(history, plot_save_path)

        if _is_main_process(context) and checkpointing_enabled:
            # Final plot
            final_plot_path = os.path.join(checkpoint_dir, "final_training_curves.png")
            plot_training_curves(history, final_plot_path)

            # Save history to file
            history_path = os.path.join(checkpoint_dir, "training_history.npy")
            np.save(history_path, history)
            print(f"Training history saved to: {history_path}")
    finally:
        _cleanup_distributed(context)

if __name__ == "__main__":
    train()
    
