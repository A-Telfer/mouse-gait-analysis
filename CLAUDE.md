# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Early scaffold: the package is `src/mouse_gait/` (src layout, `uv_build` backend, so `import mouse_gait` works after `uv sync`). `src/mouse_gait/main.py` is a placeholder exposed as the `mouse-gait` console script, and `scripts/`, `notebooks/` and `tests/` are empty. `Makefile` and `.env.template` exist but are empty, and there are no tests yet. Update this file as the pipeline takes shape.

## Lint, type check, format

ruff and mypy are dev dependencies with default settings, with two exceptions. mypy ignores missing `deeplabcut` imports. ruff targets Python 3.10 for `scripts/*`, because those run in the older-Python DLC environment, so keep scripts free of 3.11+ syntax.

- `uv run ruff check`: lint
- `uv run mypy src scripts`: type check (`deeplabcut` imports are ignored; DLC runs in its own environment)
- `uv run ruff format` and `uv run ruff check --fix`: autoformat and apply fixes

## Environment

- Managed with `uv`, Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). No dependencies declared yet.
- `uv sync` to install, `uv run mouse-gait` to run, `uv add <pkg>` to add dependencies.
- Pose-estimation tooling (e.g. DeepLabCut) may not support Python 3.14; check before adding it, and pin a lower Python if needed.

## Data

- `data/` is git-ignored (~9 GB). Never try to commit it.
- `data/raw-videos/` holds one `.mp4` per mouse per session, named `m<NN>[-<k>]_<session>.mp4`:
  - `NN` = mouse ID, `m01`–`m24`.
  - `session` is one of `acclimation`, `baseline`, `recording1`, `recording2`, `recording3`.
  - Some sessions were split into two files, `-1` and `-2` (e.g. `m17-1_recording1.mp4` + `m17-2_recording1.mp4`), with no unsuffixed file. This happens for m13, m17, m22 and m23. Filename parsing must handle the suffix and must not assume one video per mouse per session.
  - Not every mouse has every session. Don't hard-code a complete grid.
- `data/deeplabcut/` holds DLC projects, created by `scripts/00_setup-deeplabcut.py` from the DLC environment. Videos are linked, not copied. Scripts are numbered in pipeline order (`00_`, `01_`, …).

## Background

The project name and the 24 mice suggest this is a paraquat (PQ) mouse gait study. A related 2023 honours thesis ("A Novel Method of Gait Analysis Using Deep Learning to Evaluate Parkinson's Disease in Mice") used 24 C57BL/6 mice in PQ, LPS and saline groups. It filmed them from below in an open field at 100 fps, tracked them with DeepLabCut, and measured stride length and swing duration. It's unconfirmed whether these videos come from that study. Treatment-group assignments are not in the repo, so ask rather than infer them.
