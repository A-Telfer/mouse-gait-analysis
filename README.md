# mouse-gait-pq

Run `make` to list shortcuts for the commands below (`make install`, `make format`, `make check`, `make 02-deeplabcut`, …).

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

## Arena corners and cropping

Label the arena's 4 corners in every raw video:

```sh
make 00-corners   # or: uv run python scripts/00_label_arena_corners.py
```

This opens napari with one frame per video (scroll with the slider; the window title shows which video) and the polygon tool active. Click the arena's 4 corners and double-click to finish. Shift-C copies the current polygon to every video with the same frame size. The Arena review panel on the right counts the videos still to do (anything other than exactly 1 shape with 4 points), has a Next to do button that jumps to the next one (disabled when none are left), and shows a histogram of the areas of all 4-corner polygons, with a yellow line for the current video, so outliers stand out. Saved corners are loaded from `data/arena_corners.csv` at startup. Rows are keyed by video file name, never by slider position, so sort order can't misalign them; don't overwrite the file with napari's File > Save. Press Cmd/Ctrl+S to save at any time; closing the window also saves. Either way, every 4-corner polygon is written back to that file, one row per video with corners top-left, top-right, bottom-right, bottom-left.

Then crop the videos to the arena (not implemented yet):

```sh
make 01-crop   # or: uv run python scripts/01_crop_videos.py
```

## DeepLabCut setup

DeepLabCut is installed with the project (`uv sync`). Create the project with:

```sh
make 02-deeplabcut   # or: uv run python scripts/02_deeplabcut_setup.py
```

Options (all optional): `--experimenter` (default: your login name) and `--name` (default: `mouse-gait`). Pass them through make with `ARGS="--experimenter <name>"`.

This creates a project in `data/deeplabcut/` that links to (doesn't copy) every cropped video in `data/videos/cropped/`. If a project already exists there, the script asks before creating another.

Then extract frames to label:

```sh
make 03-deeplabcut-frames   # or: uv run python scripts/03_deeplabcut_extract_uniform_frames.py
```

Frames are picked completely at random from each video. Option: `--frames-per-video` (default: 5, about 600 frames across all videos). If frames already exist, the script asks before adding more.

Then label the frames:

```sh
make 04-deeplabcut-label   # or: uv run python scripts/04_deeplabcut_label_frames.py
```

This opens the newest project (by the date in its folder name) in DeepLabCut's labeling GUI.
