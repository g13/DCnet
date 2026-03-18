# DCnet Reproduction Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a paper-vs-code reproduction audit for DCnet, add targeted inline comments with tensor-dimension tracking, and verify whether `model_fig4.py` is a faithful Figure 4 reproduction path.

**Architecture:** Work from the paper and the original commit outward: first identify the active training, data, and modulation code paths; then map those paths to the paper's architecture and setup; then annotate the code and add a standalone reproduction guide. Verification is smoke-test only: import checks, config sanity, and optional single-forward-pass tests, with no full training runs.

**Tech Stack:** Python, PyTorch, Hydra, YAML configs, git history, markdown documentation.

---

### Task 1: Capture paper-critical setup differences

**Files:**
- Create: `docs/reproduction-paper-alignment.md`
- Reference: `config/config1.yaml`
- Reference: `config/config.yaml.bak`
- Reference: `README.md`

**Step 1: Write the comparison skeleton**

Create a markdown document that has sections for architecture, training hyperparameters, data/task setup, and Figure 4.

**Step 2: Fill in the paper column**

Use the paper text to record the expected setup values and behaviors.

**Step 3: Fill in the original/current commit columns**

Use git history and checked-in files to capture what `c27578b` and `d297332` actually do.

**Step 4: Add closest-paper recommendations**

For each mismatch, record whether it is `must change`, `probably change`, or `can ignore`.

**Step 5: Smoke-check the document for internal consistency**

Run: `python -m py_compile train.py model.py data.py utils.py model_fig4.py`
Expected: no syntax errors in the currently edited Python files.

### Task 2: Annotate the active config and training path

**Files:**
- Modify: `config/config1.yaml`
- Modify: `train.py`

**Step 1: Comment the config values that affect paper faithfulness**

Annotate dimensions, architecture widths, kernel sizes, modulation settings, batch/LR choices, holdout semantics, and any dead or currently unused training knobs.

**Step 2: Comment the training loop input/output semantics**

Annotate the shapes and meanings of `cue`, `mixture`, `labels`, `outputs`, and the role of `all_timesteps`.

**Step 3: Comment the scheduler/optimizer path**

Call out where config values match or diverge from the paper and whether a config field is currently unused.

**Step 4: Smoke-test imports**

Run: `python - <<'PY'
import train
print('ok')
PY`
Expected: module imports cleanly without starting training.

### Task 3: Annotate model state, dimensions, and paper alignment

**Files:**
- Modify: `model.py`

**Step 1: Comment low-rank modulation semantics**

Explain the shapes of cue activations, pooled channel vectors, outer-product spatial terms, and the modulated output tensor.

**Step 2: Comment EI cell state updates**

Annotate `input`, `h_pyr`, `h_inter`, `fb`, `exc_pyr`, `exc_inter`, `inh_pyr`, `cnm_*`, `tau_*`, and `out` with shape/meaning notes.

**Step 3: Comment the multi-layer recurrent forward pass**

Document how cue and scene are processed in two phases, how `outs_cue` is reused during the scene phase, how layer outputs are indexed by time and layer, and where the implementation does or does not match the paper.

**Step 4: Smoke-test one forward pass**

Run: `python - <<'PY'
import torch, yaml
from model import Conv2dEIRNN
cfg = yaml.safe_load(open('config/config1.yaml'))['model']
m = Conv2dEIRNN(**cfg)
cue = torch.randn(1, 3, 128, 128)
scene = torch.randn(1, 3, 128, 128)
out = m(cue, scene)
print(tuple(out.shape))
PY`
Expected: prints a logits shape compatible with `num_classes`.

### Task 4: Annotate data construction and label semantics

**Files:**
- Modify: `data.py`
- Modify: `utils.py`

**Step 1: Comment dataset JSON parsing**

Document what fields are expected in the metadata and how cue specs map to cue images and labels.

**Step 2: Comment transform and tensor ranges**

Explain pre-batch and post-batch shapes, resize behavior, and the `[-1, 1]` rescaling.

**Step 3: Comment holdout behavior**

Mark where holdout filtering is cue-based only and where that differs from broader scene-level generalization claims.

**Step 4: Smoke-test dataset module import**

Run: `python - <<'PY'
import data
print('ok')
PY`
Expected: module imports cleanly.

### Task 5: Audit and annotate the Figure 4 reproduction path

**Files:**
- Modify: `model_fig4.py`
- Modify: `docs/reproduction-paper-alignment.md`

**Step 1: Comment tensor/state semantics in `model_fig4.py`**

Annotate the modulation path, lesion helper classes, layer indexing, and any places where the file diverges from `model.py`.

**Step 2: Mark the missing Figure 4 experiment pieces inline**

Call out that the current file does not implement the evaluation sweep, per-mode reporting, or cumulative lesion protocol.

**Step 3: Record the Figure 4 verdict in the doc**

Write a dedicated section describing what the paper requires and what the current file actually does.

**Step 4: Smoke-test file syntax**

Run: `python -m py_compile model_fig4.py`
Expected: no syntax errors.

### Task 6: Final verification and handoff

**Files:**
- Modify: `docs/reproduction-paper-alignment.md`

**Step 1: Run a combined smoke test**

Run: `python -m py_compile train.py model.py data.py utils.py model_fig4.py`
Expected: all files compile.

**Step 2: Review the final doc against the edited comments**

Ensure the doc and inline comments make the same claims about dimensions, paper alignment, and Figure 4 fidelity.

**Step 3: Summarize recommended setup changes**

End the document with a prioritized list of settings to revert or keep for closest-paper reproduction.

**Step 4: Do not commit unless explicitly asked**

Leave the working tree uncommitted and report the changed files back to the user.
