# DCnet Reproduction Analysis Design

## Goal

Analyze how the current repository differs from the paper "Flexible Context-Driven Sensory Processing in Dynamical Vision Models" and from the original author commit, then annotate the active code paths so the repository itself explains those differences.

## Approved Scope

- Compare `c27578b` (original) against the current amended latest commit (`d297332`) and against the paper.
- Treat only behavior-affecting differences as reproduction-critical unless explicitly noted otherwise.
- Ignore the absolute `data.root` path difference.
- Do not run full training; use only small smoke tests if verification is needed.
- Produce both:
  - a detailed analysis document for paper-to-code alignment and reproduction guidance
  - inline comments in the key source files

## Deliverables

### 1. Reproduction analysis document

The document should include:

- a paper-to-code mapping for the main architecture and training assumptions
- a commit-diff table with columns for `paper`, `original commit`, `current commit`, and `closest-paper recommendation`
- a ranked list of reproduction risks (`must change`, `probably change`, `can ignore`)
- a dedicated Figure 4 audit section describing whether the lesion logic and evaluation protocol match the paper

### 2. Inline source annotations

Comments should be added only around behavior-defining blocks, not every line. They should explain:

- what the block does
- the dimensions and meaning of key tensors/states
- whether the implementation matches the paper, the original commit, both, or neither
- any current ambiguity or reproduction risk

The primary targets are:

- `model.py`
- `model_fig4.py`
- `train.py`
- `data.py`
- `utils.py`
- active config files in `config/`

## Figure 4 Verification Criteria

`model_fig4.py` should be evaluated against the paper's Figure 4 description, specifically:

- whether the lesion targets are modulatory synapses rather than unrelated pathways
- whether lesions are cumulative from the top sensory area to the bottom sensory area (`D -> A`)
- whether the intact network is preserved as the reference condition
- whether the effective lesion corresponds to "set modulation factor to 1"
- whether performance is reported separately for color, shape, and conjunction trials

## Verification Plan

- Prefer static analysis and commit-diff inspection.
- If code execution is needed, restrict verification to cheap smoke tests such as import checks, one forward pass with synthetic tensors, or small non-training evaluation checks.
- Do not launch long training runs.

## Expected Outcome

After the work is complete, the repository should contain:

- a readable audit of paper-vs-code differences
- annotated code paths that track tensor shapes and semantic meaning
- a clearer statement of what must be changed to approximate the paper more closely
- a specific judgment on whether the current Figure 4 reproduction path is faithful
