# Contributing to GeoAnnotate3D

Thanks for considering contributing. This document covers how the project is organized, how to get a dev environment running, and the conventions the codebase follows so a pull request doesn't need a large back-and-forth to land.

## Ways to help

- **Report bugs** — [open an issue](https://github.com/Yutlanii/GeoAnnotate3D/issues) with your OS, Python version, exact error message, and, if the app crashed, the contents of `~/.geoannotate3d/logs/crash.log`.
- **Suggest features** — open an issue describing the use case, not just the feature; it's easier to evaluate a request in context.
- **Improve the docs** — the [`docs/`](docs/) folder ([user guide](docs/user-guide.md), [tutorial](docs/tutorial.md), [FAQ](docs/faq.md)) always has room for more real-world examples.
- **Submit code** — see below.

## Project layout

```
geoannotate3d/
├── main.py              # entry point
├── core/                 # point cloud I/O, tiles, octree, terrain, project (.geoa3d) format
├── annotation/           # labeling tools, label store (undo/redo), region growing, exporter
├── render/                # VTK canvas, LOD, EDL shading, color modes
├── ui/                    # all PyQt5 panels/dialogs
│   ├── theme.py           # the ONLY place color hex values should live
│   ├── styles.py          # global QSS built from theme.py tokens
│   ├── icons.py           # loads/recolors Bootstrap Icons SVGs (ui/icons/*.svg)
│   └── main_window.py     # top-level window, header bar, navigation rail
├── infer.py               # inference logic used by both the UI and batch mode
└── tests/
    └── verify_fixes.py    # headless regression suite (see below)
```

## Setting up a dev environment

```bash
git clone https://github.com/Yutlanii/GeoAnnotate3D.git
cd GeoAnnotate3D/geoannotate3d
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python core/build_extension.py   # optional native C extension, needs a C compiler
python main.py
```

## Running the test suite

```bash
# Windows
set QT_QPA_PLATFORM=offscreen
python tests/verify_fixes.py

# Linux/Mac
QT_QPA_PLATFORM=offscreen python tests/verify_fixes.py
```

This runs headless (no visible window needed) and covers both core logic (octree queries, tile cache correctness, label store debouncing) and UI construction/behavior (panel instantiation, signal wiring, icon presence). Add a test here for any bug fix or new feature that has a testable behavior — the suite is meant to grow, not just pass.

## Conventions this codebase follows

**Colors live in `ui/theme.py`, nowhere else.** Every other `ui/*.py` file imports color constants from there (`ACCENT`, `TEXT_DIM`, `SURFACE`, etc.) instead of hardcoding hex values. If you're adding a new UI element, reuse an existing token; only add a new one to `theme.py` if none of the existing tokens fit.

**Never call `setStyleSheet()` with a bare (unselected) rule on a widget that has children.** `widget.setStyleSheet("background:X;border:Y;")` with no selector cascades those properties down to any child widget that doesn't explicitly override them — this has caused several real "double border" / invisible-text bugs in this codebase. Instead:
```python
widget.setObjectName("myWidget")
widget.setStyleSheet("QWidget#myWidget{background:X;border:Y;}")
```

**Point cloud class colors, the 3D viewport background, and gradient legends (e.g. the AGL terrain-to-canopy gradient) are data, not theme** — they're intentionally excluded from the UI color palette and should stay as explicit values, documented inline where they occur.

**Panels that hold real content should support being resized**, not `setFixedWidth()` — use `setMinimumWidth`/`setMaximumWidth` instead so the user's splitter drag actually does something. (Navigation-rail-style fixed-width elements, like the left step rail, are an intentional exception — they're chrome, not content.)

## Pull requests

- Keep PRs focused — one fix or feature per PR is easier to review than a bundle of unrelated changes.
- Run the test suite before submitting, and add a test for what you fixed if the suite doesn't already cover it.
- Match the existing code's comment density and language (comments are in Spanish throughout the codebase — please keep new ones consistent with that, even if your PR description is in English).
- Describe *why* a change was made, not just what changed, especially for bug fixes — the next person reading the diff should understand the failure mode you were fixing.

## License

By contributing, you agree your contribution is licensed under the project's [GPLv3 license](LICENSE).
