"""Shared helpers for the DeepLabCut scripts."""

import csv
import sys
import traceback
from collections.abc import Callable
from datetime import date
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEO_DIR = ROOT / "data" / "videos" / "raw"
CROPPED_VIDEO_DIR = ROOT / "data" / "videos" / "cropped"
PROJECT_DIR = ROOT / "data" / "deeplabcut"
CORNERS_CSV = ROOT / "data" / "arena_corners.csv"
CORNERS = ["tl", "tr", "br", "bl"]  # clockwise from top-left
CORNERS_COLUMNS = ["video"] + [f"{c}_{axis}" for c in CORNERS for axis in "xy"]


class Stopped(Exception):
    """Script ended early without doing its job."""


def find_projects(project_dir: Path = PROJECT_DIR) -> list[Path]:
    """Return config.yaml paths of existing projects.

    Args:
        project_dir: folder holding DLC projects
    """
    return sorted(project_dir.glob("*/config.yaml"))


def choose_project(project_dir: Path = PROJECT_DIR) -> Path:
    """Return the project's config.yaml; ask if there are several.

    Args:
        project_dir: folder holding DLC projects
    """
    projects = find_projects(project_dir)
    if not projects:
        raise Stopped(f"no project in {project_dir}; run 02_deeplabcut_setup.py first")
    if len(projects) == 1:
        return projects[0]
    print("DeepLabCut projects:")
    for i, config in enumerate(projects, 1):
        print(f"  {i}. {config.parent.name}")
    answer = input(f"Which project? [1-{len(projects)}] ").strip()
    if not answer.isdigit() or not 1 <= int(answer) <= len(projects):
        raise Stopped(f"invalid choice: {answer!r}")
    return projects[int(answer) - 1]


def latest_project(project_dir: Path = PROJECT_DIR) -> Path:
    """Return the config.yaml of the newest project, by the date in its name.

    Args:
        project_dir: folder holding DLC projects
    """
    dated = []
    for config in find_projects(project_dir):
        try:  # DLC names projects <name>-<experimenter>-YYYY-MM-DD
            day = date.fromisoformat(config.parent.name[-10:])
        except ValueError:
            continue
        dated.append((day, config.stat().st_mtime, config))
    if not dated:
        raise Stopped(f"no project in {project_dir}; run 02_deeplabcut_setup.py first")
    return max(dated)[2]


def load_corners(path: Path) -> dict[str, list[list[float]]]:
    """Return saved [x, y] corners by video file name, or {} if there's no file.

    Rows are keyed by video name, never by position, so sort order can't
    misalign them. Corners with a blank or non-numeric x or y are left out, so
    a video can have fewer than 4.

    Args:
        path: corners CSV
    """
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if rows and "video" not in rows[0]:
        raise Stopped(f"{path} has no 'video' column; not a corners file")
    corners = {}
    for row in rows:
        if row.get("video"):
            points = []
            for c in CORNERS:
                try:
                    points.append([float(row[f"{c}_x"]), float(row[f"{c}_y"])])
                except (KeyError, TypeError, ValueError):
                    continue
            corners[row["video"]] = points
    return corners


def confirm(prompt: str) -> bool:
    """Ask a yes/no question; default no."""
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def import_deeplabcut() -> ModuleType:
    """Import DeepLabCut, or stop with a hint if it isn't installed."""
    try:
        import deeplabcut
    except ModuleNotFoundError:
        raise Stopped("DeepLabCut not installed; run uv sync")
    return deeplabcut


def run(main: Callable[[], str], failed: str) -> None:
    """Run main; print SUCCESS with its message, or WARNING on any other exit.

    Args:
        main: returns the success message
        failed: what didn't happen, e.g. "no project created"
    """
    try:
        message = main()
    except Stopped as e:
        sys.exit(f"WARNING: {failed}: {e}")
    except (KeyboardInterrupt, EOFError):
        sys.exit(f"\nWARNING: {failed}: cancelled")
    except Exception:  # noqa: BLE001 - any crash ends in a warning
        traceback.print_exc()
        sys.exit(f"WARNING: {failed}: unexpected error (see above)")
    print(f"SUCCESS: {message}")
