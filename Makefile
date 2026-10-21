.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck check 00-corners 01-crop 02-deeplabcut 03-deeplabcut-frames 04-deeplabcut-label 05-deeplabcut-arena

help: ## list available commands
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-21s\033[0m %s\n", $$1, $$2}'

install: ## install dependencies (uv sync)
	uv sync

format: ## autoformat and apply lint fixes
	uv run ruff format
	uv run ruff check --fix
	uv run rumdl check --fix .

lint: ## lint Python with ruff and markdown with rumdl
	uv run ruff check
	uv run rumdl check .

typecheck: ## type check with mypy
	uv run mypy src scripts

check: lint typecheck ## lint and type check

00-corners: ## label the arena's 4 corners in every video (napari)
	uv run python scripts/00_label_arena_corners.py

01-crop: ## crop videos to the arena (no re-encoding)
	uv run python scripts/01_crop_videos.py

02-deeplabcut: ## create DeepLabCut project (optional ARGS="--experimenter <name>")
	uv run python scripts/02_deeplabcut_setup.py $(ARGS)

03-deeplabcut-frames: ## extract frames to label (optional ARGS="--frames-per-video 5")
	uv run python scripts/03_deeplabcut_extract_uniform_frames.py $(ARGS)

04-deeplabcut-label: ## open newest project in the labeling GUI
	uv run python scripts/04_deeplabcut_label_frames.py

05-deeplabcut-arena: ## mark arena corners in the newest DeepLabCut project (napari)
	uv run python scripts/05_deeplabcut_mark_arena.py
