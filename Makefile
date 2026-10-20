.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck check setup-dlc

help: ## list available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

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

setup-dlc: ## create DeepLabCut project (run in the DLC env; ARGS="--experimenter <name>")
	python scripts/00_setup-deeplabcut.py $(ARGS)
