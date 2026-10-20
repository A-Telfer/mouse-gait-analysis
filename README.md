# mouse-gait-pq

Run `make` to list shortcuts for the commands below (`make install`, `make format`, `make check`, `make 00-deeplabcut`, …).

## Setup

```sh
uv sync
```

## Linting and type checking

```sh
uv run ruff check   # lint
uv run mypy src scripts  # type check
```

## Autoformatting

```sh
uv run ruff format       # format code
uv run ruff check --fix  # apply auto-fixable lint fixes
```

## DeepLabCut setup

DeepLabCut is installed with the project (`uv sync`). Create the project with:

```sh
make 00-deeplabcut   # or: uv run python scripts/00_setup-deeplabcut.py
```

Options (all optional): `--experimenter` (default: your login name) and `--name` (default: `mouse-gait`). Pass them through make with `ARGS="--experimenter <name>"`.

This creates a project in `data/deeplabcut/` that links to (doesn't copy) every video in `data/raw-videos/`. If a project already exists there, the script asks before creating another.

Then extract frames to label:

```sh
make 01-deeplabcut-frames   # or: uv run python scripts/01_deeplabcut_extract_initial_frames.py
```

Frames are picked completely at random from each video. Option: `--frames-per-video` (default: 5, about 600 frames across all videos). If frames already exist, the script asks before adding more.
