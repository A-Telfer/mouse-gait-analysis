# mouse-gait-pq

## Setup

```sh
uv sync
```

## Linting and type checking

```sh
uv run ruff check   # lint
uv run mypy src     # type check
```

## Autoformatting

```sh
uv run ruff format       # format code
uv run ruff check --fix  # apply auto-fixable lint fixes
```
