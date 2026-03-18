# Model Diagram Export Design

## Goal

Make the saved Mermaid model diagram readable by default when `train.py` writes it to disk.

## Root Cause

- The current export path in `utils.py` always writes a `.png` file.
- The Mermaid CLI invocation uses its default raster settings, which produced a very small image in practice (`784x105` in the current worktree).
- The diagram content itself is fine; the unreadability comes from the export format and raster size.

## Approved Direction

- Save the diagram as a PDF by default instead of PNG.
- Keep the code change tight to the existing export path instead of refactoring unrelated reporting logic.
- Update tests so the saved filename and renderer behavior cover the new default.

## Design

### Export behavior

- Introduce a format-aware diagram save helper in `utils.py`.
- Default the saved artifact extension to `.pdf`.
- Continue to use Mermaid CLI (`mmdc`) as the renderer so the implementation stays aligned with the existing toolchain.

### Training entry point

- Update `train.py` to import and call the format-aware save helper.
- Change the user-facing log messages from `PNG`-specific wording to generic `model diagram` wording.

### Backward compatibility

- Keep a PNG-specific wrapper available in `utils.py` so existing call sites or tests can still request PNG output explicitly if needed.
- Preserve the existing `save_model_diagram_png` config switch for now to avoid widening the scope of this bugfix.

## Testing

- Add a regression test that proves the default saved filename now ends in `.pdf`.
- Keep coverage for explicit PNG output so the compatibility path still works.
- Run the focused parameter-reporting test module after the code change.
