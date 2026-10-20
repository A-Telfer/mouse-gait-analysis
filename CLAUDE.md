# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Early scaffold: `src/main.py` is a placeholder, and `scripts/`, `notebooks/` and `tests/` are empty. `Makefile`, `README.md` and `.env.template` exist but are empty, so there are no build, lint or test commands yet. Update this file as the pipeline takes shape.

## Environment

- Managed with `uv`, Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). No dependencies declared yet.
- `uv sync` to install, `uv run python src/main.py` to run, `uv add <pkg>` to add dependencies.
- Pose-estimation tooling (e.g. DeepLabCut) may not support Python 3.14; check before adding it, and pin a lower Python if needed.

## Data

- `data/` is git-ignored (~9 GB). Never try to commit it.
- `data/raw-videos/` holds one `.mp4` per mouse per session, named `m<NN>[-<k>]_<session>.mp4`:
  - `NN` = mouse ID, `m01`–`m24`.
  - `session` is one of `acclimation`, `baseline`, `recording1`, `recording2`, `recording3`.
  - Some sessions were split into two files, `-1` and `-2` (e.g. `m17-1_recording1.mp4` + `m17-2_recording1.mp4`), with no unsuffixed file. This happens for m13, m17, m22 and m23. Filename parsing must handle the suffix and must not assume one video per mouse per session.
  - Not every mouse has every session. Don't hard-code a complete grid.

## Background

The project name and the 24 mice suggest this is a paraquat (PQ) mouse gait study. A related 2023 honours thesis ("A Novel Method of Gait Analysis Using Deep Learning to Evaluate Parkinson's Disease in Mice") used 24 C57BL/6 mice in PQ, LPS and saline groups. It filmed them from below in an open field at 100 fps, tracked them with DeepLabCut, and measured stride length and swing duration. It's unconfirmed whether these videos come from that study. Treatment-group assignments are not in the repo, so ask rather than infer them.
