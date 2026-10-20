# mouse-gait-pq

Run `make` to list shortcuts for the commands below (`make install`, `make format`, `make check`, `make setup-dlc`, …).

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

DeepLabCut needs its own environment (it doesn't support this project's Python 3.14). From that environment, run:

```sh
python scripts/00_setup-deeplabcut.py --experimenter <name>
```

This creates a project in `data/deeplabcut/` that links to (doesn't copy) every video in `data/raw-videos/`. If a project already exists there, the script asks before creating another.
