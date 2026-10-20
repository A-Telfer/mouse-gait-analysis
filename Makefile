.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck check 00-deeplabcut 01-deeplabcut-frames

help: ## list available commands
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-21s\033[0m %s\n", $$1, $$2}'

install: ## install dependencies (uv sync)
	uv sync

format: ## autoformat and apply lint fixes
	uv run ruff format
	uv run ruff check --fix

lint: ## lint with ruff
	uv run ruff check

typecheck: ## type check with mypy
	uv run mypy src scripts

check: lint typecheck ## lint and type check

00-deeplabcut: ## create DeepLabCut project (run in the DLC env; optional ARGS="--experimenter <name>")
	python scripts/00_setup-deeplabcut.py $(ARGS)

01-deeplabcut-frames: ## extract frames to label (run in the DLC env; optional ARGS="--frames-per-video 5")
	python scripts/01_deeplabcut_extract_initial_frames.py $(ARGS)
