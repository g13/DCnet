# DCnet-2026.2.13

## Forward-pass data flow (DCnet / `Conv2dEIRNN`)

```mermaid
flowchart TD
    A["Input batch from dataloader<br/>tuple: cue, mixture, label<br/>`for i, (cue, mixture, labels) in train_loader`<br/>`train_new2.py` L70"] --> B["Call model forward<br/>`outputs = model(cue, mixture, ...)`<br/>`train_new2.py` L76"]

    B --> C["Enter `Conv2dEIRNN.forward(...)`<br/>`model.py` L727-L734"]
    C --> D["Two-stage stimulation loop<br/>`for stimulation in (cue, mixture)`<br/>`model.py` L756"]

    D --> E["Initialize hidden / fb / outputs for this stage<br/>`_init_hidden` L766-L768, `_init_fb` L770-L772, `_init_out` L775-L777<br/>`model.py`"]
    E --> F["Time loop and layer loop<br/>`for t in range(self.num_steps)` L778<br/>`for i, layer in enumerate(self.layers)` L787<br/>`model.py`"]

    subgraph L[Per-step and per-layer computation with exact calls]
      F --> G["Compute one EI layer update<br/>`layer(...)` call in `Conv2dEIRNN.forward`<br/>`model.py` L806-L819"]
      G --> H["Inside `Conv2dEIRNNCell.forward(...)`<br/>`model.py` L334-L397"]
      H --> H1["Excitation path<br/>`conv_exc_pyr(torch.cat(...))` + pre_inh activation<br/>`model.py` L358-L363"]
      H1 --> H2["Interneuron/inhibition path (if enabled)<br/>`conv_exc_inter(...)` L367-L372 then `conv_inh(...)` L374<br/>`model.py`"]
      H2 --> H3["Candidate states<br/>`cnm_pyr = post_inh_activation(exc_pyr - inh_pyr)`<br/>`model.py` L379"]
      H3 --> H4["Euler membrane update with learnable tau<br/>`tau_pyr = sigmoid(self.tau_pyr)` L385<br/>`h_next_pyr = (1-tau)*h + tau*cnm` L386<br/>`model.py`"]
      H4 --> H5["Layer output pooling<br/>`out = self.out_pool(h_next_pyr)`<br/>`model.py` L395"]
      H5 --> I["Optional cue-driven modulation on mixture stage<br/>condition + `self.modulations[i](out_cue, outs[t][i])`<br/>`model.py` L822-L843<br/>`LowRankModulation.forward` `model.py` L27-L45"]
      I --> J["Optional feedback accumulation<br/>`fbs[t][j] += self.fb_convs[...] (outs[t][i])`<br/>`model.py` L845-L851"]
    end

    J --> K["After cue stage: cache cue activations<br/>`outs_cue = outs`; `h_pyrs_cue = h_pyrs`<br/>`model.py` L865-L867"]
    K --> M["Readout selection<br/>final step: `out = self.out_layer(outs[-1][-1])` L874<br/>or all timesteps at L869-L873<br/>`model.py`"]
    M --> N["Classification head (`self.out_layer`)<br/>Flatten -> Linear -> ReLU -> Dropout -> Linear<br/>`model.py` L667-L677"]
    N --> O["Training objective and metrics<br/>`loss = criterion(outputs, labels)` L84<br/>`predicted = outputs.argmax(-1)` L98<br/>`train_new2.py`"]
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
