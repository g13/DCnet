# DCnet-2026.2.13

## Forward-pass data flow (DCnet / `Conv2dEIRNN`)

```mermaid
flowchart TD
    A[Batch from dataloader\ncue, mixture, label] --> B[Cue pass through recurrent EI stack\nnum_steps x num_layers]
    B --> C[Store cue states\nouts_cue, h_pyrs_cue, h_inters_cue]
    C --> D[Mixture pass through same EI stack\nre-initialize hidden if flush_hidden=True]

    subgraph L[Per-step, per-layer computation]
      E[input to layer i\n(raw image for i=0, previous layer output otherwise)] --> F[Excitatory update\nconv_exc_* + pre_inh_activation]
      F --> G[Interneuron branch\nconv_exc_inter -> conv_inh]
      G --> H[Candidate state\ncnm_pyr = post_inh(exc - inh)]
      H --> I[Euler integration with learnable tau\nh_next = (1-tau)h + tau*cnm]
      I --> J[AvgPool output]
      J --> K[Optional cue-driven low-rank modulation\nouts = outs * (1 + rank1*0.1)]
      K --> M[Optional feedback to lower layers]
    end

    D --> N[Take final top-layer output\n(outs[-1][-1] or all steps)]
    N --> O[Head: flatten + FC + dropout + FC]
    O --> P[Class logits]
    P --> Q[argmax during metrics\nCrossEntropyLoss during training]
```

## Detailed training process (from `train_new2.py`)

1. **Configuration and setup**
   - Hydra loads the config, converts it to `AttrDict`, optionally seeds RNGs, and sets matmul precision.
   - Model is created as `Conv2dEIRNN(**config.model)` on CUDA, then optionally passed through `torch.compile`.

2. **Optimizer / criterion / scheduler initialization**
   - Optimizer can be SGD / Adam / AdamW (config-driven).
   - Criterion is cross-entropy (`torch.nn.CrossEntropyLoss`).
   - Scheduler supports OneCycleLR with `total_steps = epochs * len(train_loader)`.

3. **Data pipeline**
   - `get_qclevr_dataloaders(...)` returns train/val loaders of `(cue, mixture, label)` tuples.
   - The script prints dataset and first-batch class distributions before training, then dataset sizes.

4. **Epoch loop**
   - For each epoch:
     - Run `train_iter(...)` on the train loader.
     - Run `eval_iter(...)` on the validation loader.
     - Append `train_loss/train_acc/test_loss/test_acc` to in-memory history.
     - Print epoch summary.
     - Save checkpoint with model state, optimizer state, epoch, and history.
     - Periodically plot and save training curves.

5. **Inside one training iteration (`train_iter`)**
   - Set model to `train()`.
   - For each batch:
     - Move cue/mixture/labels to device.
     - Forward call: `outputs = model(cue, mixture, all_timesteps=...)`.
     - Loss:
       - If `all_timesteps=True`, compute CE for each timestep output and average.
       - Else CE on final logits.
     - Backprop:
       - `optimizer.zero_grad()`
       - `loss.backward()`
       - Gradient clipping (norm or value, depending on config)
       - `optimizer.step()`
       - `scheduler.step()` (if enabled)
     - Metrics:
       - `pred = outputs.argmax(-1)` for batch accuracy
       - running and epoch-level loss/accuracy are accumulated
     - Logging:
       - every `log_freq` batches, log running metrics to W&B (or no-op logger)

6. **Validation iteration (`eval_iter`)**
   - Set model to `eval()` and wrap with `torch.no_grad()`.
   - Forward pass is the same API as training.
   - Compute loss and argmax accuracy across validation loader.
   - Log `test_loss` and `test_acc`.

7. **Artifacts and outputs**
   - Checkpoints are saved every epoch (`checkpoint_{epoch}.pt` + copied latest `checkpoint.pt`).
   - Training curves are saved as images.
   - Final history dictionary is saved as `training_history.npy`.

## Notes specific to this implementation

- DCnet performs a **two-stage forward** in one call: cue first, then mixture. Cue activations are reused to modulate mixture activations when modulation is enabled.
- Core dynamics are recurrent EI updates with learnable membrane constants (`tau_*`) and optional feedback adjacency.
- In the default config, modulation is enabled (`modulation_type: lr`, `modulation_on: layer_output`) while perturbation is disabled.
