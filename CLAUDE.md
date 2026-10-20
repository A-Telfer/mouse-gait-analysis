# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Early scaffold: the package is `src/mouse_gait/` (src layout, `uv_build` backend, so `import mouse_gait` works after `uv sync`). `src/mouse_gait/main.py` is a placeholder exposed as the `mouse-gait` console script, and `scripts/`, `notebooks/` and `tests/` are empty. `.env.template` is empty, and there are no tests yet. `make` lists the shortcut targets; add a `## description` comment to any new target so it appears in that list. Update this file as the pipeline takes shape.

## Lint, type check, format

ruff and mypy are dev dependencies with default settings, with two exceptions. mypy ignores missing `deeplabcut` type stubs. ruff targets Python 3.10 for `scripts/*`, left over from when DLC ran in a separate older-Python environment.

- `uv run ruff check`: lint
- `uv run mypy src scripts`: type check
- `uv run ruff format` and `uv run ruff check --fix`: autoformat and apply fixes

## Environment

- Managed with `uv`, Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). DeepLabCut 3.0.2 (`deeplabcut[gui,pytorch]`) is a project dependency.
- `uv sync` to install, `uv run mouse-gait` to run, `uv add <pkg>` to add dependencies.
- DLC officially supports only Python 3.10–3.12, but it installs and imports on 3.14. If DLC errors look version-related, suspect that first. `filterpy` (a DLC dependency) prints a harmless `SyntaxWarning` the first time it's imported.

## Data

- `data/` is git-ignored (~9 GB). Never try to commit it.
- `data/raw-videos/` holds one `.mp4` per mouse per session, named `m<NN>[-<k>]_<session>.mp4`:
  - `NN` = mouse ID, `m01`–`m24`.
  - `session` is one of `acclimation`, `baseline`, `recording1`, `recording2`, `recording3`.
  - Some sessions were split into two files, `-1` and `-2` (e.g. `m17-1_recording1.mp4` + `m17-2_recording1.mp4`), with no unsuffixed file. This happens for m13, m17, m22 and m23. Filename parsing must handle the suffix and must not assume one video per mouse per session.
  - Not every mouse has every session. Don't hard-code a complete grid.
- `data/deeplabcut/` holds DLC projects, created by `scripts/00_setup-deeplabcut.py`. Videos are linked, not copied. Scripts are numbered in pipeline order (`00_`, `01_`, …). DLC scripts share helpers in `scripts/_dlc_common.py`: finding the project, yes/no prompts, and `run()`, which prints SUCCESS or WARNING. Keep new DLC scripts on that pattern, and run them with `uv run python` (the `make` targets do this).

## Background

The project name and the 24 mice suggest this is a paraquat (PQ) mouse gait study. A related 2023 honours thesis ("A Novel Method of Gait Analysis Using Deep Learning to Evaluate Parkinson's Disease in Mice") used 24 C57BL/6 mice in PQ, LPS and saline groups. It filmed them from below in an open field at 100 fps, tracked them with DeepLabCut, and measured stride length and swing duration. It's unconfirmed whether these videos come from that study. Treatment-group assignments are not in the repo, so ask rather than infer them.
