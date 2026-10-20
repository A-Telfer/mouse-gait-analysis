"""Shared helpers for the DeepLabCut scripts (run in the DLC environment)."""

import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "raw-videos"
PROJECT_DIR = ROOT / "data" / "deeplabcut"


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
        raise Stopped(f"no project in {project_dir}; run 00_setup-deeplabcut.py first")
    if len(projects) == 1:
        return projects[0]
    print("DeepLabCut projects:")
    for i, config in enumerate(projects, 1):
        print(f"  {i}. {config.parent.name}")
    answer = input(f"Which project? [1-{len(projects)}] ").strip()
    if not answer.isdigit() or not 1 <= int(answer) <= len(projects):
        raise Stopped(f"invalid choice: {answer!r}")
    return projects[int(answer) - 1]


def confirm(prompt: str) -> bool:
    """Ask a yes/no question; default no."""
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def import_deeplabcut() -> ModuleType:
    """Import DeepLabCut, or stop with a hint if it isn't installed."""
    try:
        import deeplabcut
    except ModuleNotFoundError:
        raise Stopped("DeepLabCut not installed; run from the DLC environment")
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
