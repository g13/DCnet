# Model Diagram Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make saved Mermaid model diagrams readable by default by switching the artifact written from training from low-resolution PNG output to PDF output.

**Architecture:** Keep the existing Mermaid export flow in place, but make the save helper format-aware instead of hardcoding `.png`. Use PDF as the default output for the training path, preserve an explicit PNG compatibility wrapper, and verify the change with focused regression tests around filename generation and save behavior.

**Tech Stack:** Python, unittest, Mermaid CLI (`mmdc`), pathlib.

---

### Task 1: Lock in the new default with a failing test

**Files:**
- Modify: `tests/test_parameter_reporting.py`
- Reference: `utils.py`

**Step 1: Write the failing test**

Add a test that expects `make_model_diagram_filename(date(2026, 3, 17), "abc1234")` to return `flow_chart-2026-03-17-abc1234.pdf`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_parameter_reporting.ParameterReportingTests.test_make_model_diagram_filename_uses_date_and_commit`
Expected: FAIL because the helper still returns `.png`.

**Step 3: Write a second failing save-path regression test**

Add a test that calls the default save helper and expects the output filename to end in `.pdf`.

**Step 4: Run the targeted tests to verify they fail for the expected reason**

Run: `python -m unittest tests.test_parameter_reporting.ParameterReportingTests.test_make_model_diagram_filename_uses_date_and_commit tests.test_parameter_reporting.ParameterReportingTests.test_save_mermaid_diagram_writes_pdf_by_default`
Expected: FAIL because the current save path is still PNG-only.

### Task 2: Implement format-aware diagram saving

**Files:**
- Modify: `utils.py`
- Test: `tests/test_parameter_reporting.py`

**Step 1: Update filename generation**

Extend `make_model_diagram_filename` so it accepts an output format or suffix and defaults to `pdf`.

**Step 2: Add the minimal format-aware save helper**

Implement a generic save helper that creates the output path using the chosen format and forwards the Mermaid body to the renderer.

**Step 3: Preserve explicit PNG support**

Keep a thin `save_mermaid_diagram_png(...)` wrapper that routes through the generic save helper with `png` so explicit PNG requests still work.

**Step 4: Run the focused test module**

Run: `python -m unittest tests.test_parameter_reporting`
Expected: PASS.

### Task 3: Switch the training path to the new default artifact

**Files:**
- Modify: `train.py`
- Reference: `utils.py`

**Step 1: Update imports and call site**

Replace the PNG-specific helper import with the generic helper and call it from the model-diagram save path.

**Step 2: Update user-facing messages**

Change the save and warning messages so they refer to a model diagram file instead of PNG specifically.

**Step 3: Smoke-check the training module**

Run: `python -m py_compile train.py utils.py tests/test_parameter_reporting.py`
Expected: no syntax errors.

### Task 4: Final focused verification

**Files:**
- Modify: `tests/test_parameter_reporting.py`
- Modify: `train.py`
- Modify: `utils.py`

**Step 1: Run the regression tests fresh**

Run: `python -m unittest tests.test_parameter_reporting`
Expected: PASS with the PDF default covered by tests.

**Step 2: Optionally exercise the export path manually**

Run: `python train.py output_model_structure_only=True save_model_diagram_png=True`
Expected: a `flow_chart-<date>-<commit>.pdf` file is written and the printed path points to that file.

**Step 3: Do not commit unless explicitly asked**

Leave the changes uncommitted and report the edited files and verification evidence back to the user.
