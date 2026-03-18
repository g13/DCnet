# DCnet Paper Alignment Audit

This note compares the paper setup to the original public repo commit `c27578b` and the current amended latest commit `d297332`. It focuses on behavior-affecting differences that can change reproduction outcomes.

## Executive View

- The current code path is closer to the paper than the original config in one important respect: it uses `modulation_type: lr`, which matches the paper's low-rank modulation story.
- Appendix A.2 is actually closer to the current config than to the original public config on layer widths and kernel sizes.
- The remaining important mismatches are elsewhere: smaller batch size, higher learning rate, disabled holdout split, enabled gradient clipping, an ambiguous `T` between paper text and code, and a readout/parameter-count description in the paper that does not line up cleanly with the code.
- The original config is not directly runnable against the current `model.py` because it requests `modulation_type: ag`, while the model code only accepts `lr`.
- `model_fig4.py` is not yet a faithful Figure 4 reproduction path. It contains pieces of a lesion idea, but not the cumulative lesion protocol or the required evaluation/data collection.

## Paper To Code Mapping

### Architecture

- Paper: 4 sensory areas with excitatory/inhibitory populations, 4:1 E/I ratio, stride-2 pooling between areas, low-rank cue-driven modulation of pooled excitatory activity.
- Code path: `model.py` implements 4 recurrent convolutional areas, with pooled excitatory outputs passed forward and optional feedback via `fb_adjacency`.
- Match status: mostly aligned at a high level.

### Cue Then Scene Protocol

- Paper: cue for `T` steps, then scene for `T` steps, with readout at the final time.
- Code path: `Conv2dEIRNN.forward()` iterates over `(cue, mixture)` and unrolls each for `num_steps=5`.
- Match status: conceptually aligned, but not numerically pinned down by the paper text. Appendix B explicitly says `T=3` for the Convolutional RNN baseline; the repo's DCnet uses `T=5`.

### Low-Rank Modulation

- Paper: pooled excitatory activity is projected to two vectors, outer-product structure creates a low-rank modulation factor, and that factor multiplicatively modulates the sensory response.
- Code path: `LowRankModulation` spatially averages the cue tensor to `[B, C_l]`, projects to `H_l` and `W_l`, builds a rank-1 spatial map, broadcasts it back to `[B, C_l, H_l, W_l]`, and multiplies the current pooled output.
- Match status: qualitatively aligned, but the exact factorization differs from the Appendix notation and the active branch uses a harsher direct multiplication than the residual-style variant left commented in `model.py`.

## Reproduction-Critical Comparison

| Item | Paper | Original `c27578b` | Current `d297332` | Closest-paper recommendation |
| --- | --- | --- | --- | --- |
| Sensory areas | 4 | 4 | 4 | keep current |
| Input resolution | 128x128 after resize | 128x128 | 128x128 | keep current |
| Cue then scene timing | `T` then `T` | `num_steps: 5` | `num_steps: 5` | keep current |
| Pyramidal widths | Appendix A.2 is closer to `[16, 32, 64, 128]` | `[16, 32, 64, 64]` | `[16, 32, 64, 128]` | keep current widths if you follow A.2, but note the conflict with the paper's ~1.8M parameter claim |
| Interneuron widths | 4:1 ratio; Appendix A.2 is closer to `[4, 8, 16, 32]` | `[4, 8, 16, 16]` | `[4, 8, 16, 32]` | keep current widths if you follow A.2 |
| Kernel sizes | Appendix A.2 is closer to `[5,5],[5,5],[5,5],[3,3]` | `[5,5],[5,5],[3,3],[3,3]` | `[5,5],[5,5],[5,5],[3,3]` | keep current kernels if you follow A.2 |
| Modulation type | low-rank | `ag` in config | `lr` in config | keep `lr`; the original config is inconsistent with current code |
| Modulation target | pooled excitatory output | `layer_output` | `layer_output` | keep current |
| Batch size | 256 | 256 | 128 | revert to 256 |
| Optimizer | AdamW | AdamW | AdamW | keep current |
| Max learning rate | `4e-4` | `4e-4` | `1e-3` | revert to `4e-4` |
| Scheduler | one-cycle, `pct_start=0.3` | yes | yes, but `pct_start` unused in code | pass `pct_start` explicitly or rely on the library default knowingly |
| Epochs | 100 in appendix | 500 | 100 | keep 100 |
| Gradient clipping | not reported | disabled | enabled | disable for paper closeness |
| Holdout cues | used for harder generalization tests | `[blue, green]` | `[]` | restore holdout cues when reproducing generalization |
| Dataset path | external | `data/qclevr` | custom local path | ignore absolute path; only dataset semantics matter |
| `T` / steps per phase | Appendix B explicitly says `T=3` for the baseline; DCnet text is ambiguous | `num_steps: 5` | `num_steps: 5` | treat this as an ambiguity; the repo DCnet uses 5 |
| Readout size | paper text says `(1024 x 6)` | code uses `4096 -> 256 -> 6` | code uses `8192 -> 256 -> 6` | treat the paper's readout statement as another text-vs-code inconsistency |

## Parameter Count Snapshot

- Current config (`config/config1.yaml`): about `3,497,630` parameters.
- Original-width config (`config/config.yaml.bak` with `modulation_type` manually switched to `lr` so it can instantiate): about `1,990,158` parameters.
- Paper target: about `1.8M` parameters.

This creates a real inconsistency inside the paper/package story: Appendix A.2 points toward the current wider config, but the separate `~1.8M` parameter claim is much closer to the smaller original-width config.

## Important Code-Level Divergences

### 1. Original config is internally inconsistent

- `config/config.yaml.bak` requests `modulation_type: ag`.
- `model.py` rejects any modulation type other than `lr`.
- Consequence: the original config is not directly runnable against the current model implementation without editing the config.

### 2. `pct_start` is effectively dead config

- `config/config1.yaml` defines `scheduler.pct_start: 0.3`.
- `train.py` does not pass `pct_start` into `OneCycleLR`.
- Consequence: the behavior matches the PyTorch default only as long as that default remains `0.3`.

### 3. `accumulation_steps` is not implemented

- `config/config1.yaml` sets `accumulation_steps: 4`.
- `train.py` still performs `zero_grad -> backward -> step` every batch.
- Consequence: effective batch size is still 128, not 512 and not paper-like 256.

### 4. Holdout logic is cue-only

- `data.py` filters by cue identity only.
- There is no scene-level holdout mechanism in the loader.
- Consequence: cue-holdout generalization can be tested, but broader "novel scenes and novel cues" claims require additional split logic outside the current loader.

### 5. Hidden-state handling matters

- `flush_hidden: True` resets recurrent hidden states before the scene phase.
- Cue information still reaches the scene phase through cached cue activations used for modulation.
- Consequence: the implementation is still cue-conditioned, but not via literal hidden-state carryover from cue to scene.

## Dataset Notes

- The loader supports `color`, `shape`, and `conjunction` trials.
- The active config uses only `mode: color`, so it cannot reproduce the full paper suite by itself.
- Labels are count classes `0..5`, consistent with `num_classes: 6`.
- If you point the code at a local dataset variant whose object-count range differs from the paper's 3-10 objects, that dataset drift matters more than the absolute path string.

## Figure 4 Audit

### What the paper says

Figure 4 requires a trained DCnet to be evaluated under cumulative lesions of the modulatory synapses, starting from the top sensory area (`D`) and moving down to the bottom sensory area (`A`), while recording task performance separately for color, shape, and conjunction trials. The intact network is reported as the reference condition.

### What `model_fig4.py` currently does

- It defines an alternate `LowRankModulation` that applies a residual/gain-style factor `mixture * (1 + 0.1 * M)`.
- It defines `LowRankModulation_mute`, which would behave like a lesion by returning the unmodulated signal.
- It never wires `LowRankModulation_mute` into the active modulation list.
- It modulates each layer's scene output with the same layer's cue output (`outs_cue[t][i] -> outs[t][i]`).
- It is not used by `train.py`, which still imports `Conv2dEIRNN` from `model.py`.

### Verdict

`model_fig4.py` is not a faithful Figure 4 reproduction path yet.

### Why it falls short

- No cumulative lesion sweep `D -> C -> B -> A` is implemented.
- No intact-vs-lesioned evaluation harness is implemented.
- No per-mode reporting for color, shape, and conjunction is implemented.
- The lesion helper exists only as an unused class.
- The modulation path is still same-layer cue-to-scene modulation, not an explicit higher-order-module-to-sensory-area lesion interface.

### What would be needed for a closer Figure 4 match

- A checkpoint-loading eval script that explicitly imports `model_fig4.py` or merges the lesion logic into `model.py`.
- A layer selection mechanism that replaces the modulation factor with exactly `1` for lesioned areas.
- A cumulative lesion schedule using layer index `3 -> 2 -> 1 -> 0` to represent paper areas `D -> C -> B -> A`.
- Separate evaluation loops or bookkeeping for `color`, `shape`, and `conjunction` trials.

## Ranked Recommendations

### Must change

- Restore `batch_size: 256`.
- Restore `optimizer.lr: 0.0004`.
- Disable gradient clipping for paper-faithful runs.
- Restore cue holdouts when reproducing the generalization results.

### Probably change

- Decide whether to trust Appendix A.2 widths/kernels or the paper's separate `~1.8M` and `(1024 x 6)` statements when these disagree.
- Pass `pct_start` explicitly to `OneCycleLR` so the config controls the scheduler instead of silently relying on library defaults.
- Remove or implement `accumulation_steps` so the config reflects reality.
- Decide whether DCnet should use `T=3` or `T=5`; the repo uses `5`, while Appendix B only explicitly states `3` for the baseline.
- Add a dedicated Figure 4 evaluation script instead of relying on `model_fig4.py` alone.

### Can ignore

- The absolute `data.root` path string, as long as the dataset behind that path matches the paper semantics.
- Extra logging, plotting, and checkpoint-copy conveniences in `train.py`, since they do not materially change the model definition.

## Suggested Closest-Paper Starting Point

Start from the current branch, but combine these choices:

- keep `modulation_type: lr`
- keep `modulation_on: layer_output`
- keep Appendix A.2 widths/kernels if you prioritize the architecture table
- keep `num_layers: 4` and `epochs: 100`
- decide explicitly between `num_steps: 5` (repo DCnet) and `T=3` (only explicit in Appendix B baseline text)
- revert learning settings closer to the appendix values (`batch_size: 256`, `lr: 4e-4`, holdouts enabled, clipping disabled)
- restore holdout cues when testing generalization
- treat Figure 4 as a separate evaluation implementation task, because the current `model_fig4.py` does not complete that experiment
