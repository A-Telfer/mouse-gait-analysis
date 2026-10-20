"""Create a DeepLabCut project in data/deeplabcut from the videos in data/raw-videos.

Run with the Python environment that has DeepLabCut installed.
"""

import argparse
import getpass
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "raw-videos"
PROJECT_DIR = ROOT / "data" / "deeplabcut"
BODYPARTS = [
    "nose",
    "chest",
    "tailbase",
    "midtail",
    "tail_tip",
    "left_front_paw",
    "right_front_paw",
    "left_toes",
    "left_back_paw",
    "right_toes",
    "right_back_paw",
]


class SetupStopped(Exception):
    """Setup ended without creating a project."""


def find_videos(video_dir: Path) -> list[str]:
    """Return paths of all .mp4 videos; stop if any are missing.

    Args:
        video_dir: folder of raw videos
    """
    if not video_dir.is_dir():
        raise SetupStopped(f"video folder not found: {video_dir}")
    paths = sorted(video_dir.glob("*.mp4"))
    missing = [p for p in paths if not p.is_file()]  # e.g. broken links
    if missing:
        listing = "\n".join(f"  {p}" for p in missing)
        raise SetupStopped(f"unreadable videos:\n{listing}")
    if not paths:
        raise SetupStopped(f"no .mp4 videos found in {video_dir}")
    return [str(p) for p in paths]


def find_projects(project_dir: Path) -> list[Path]:
    """Return config.yaml paths of existing projects.

    Args:
        project_dir: folder holding DLC projects
    """
    return sorted(project_dir.glob("*/config.yaml"))


def confirm(prompt: str) -> bool:
    """Ask a yes/no question; default no."""
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def main() -> str:
    """Create the project and return its config.yaml path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default="mouse-gait", help="project name (default: mouse-gait)"
    )
    parser.add_argument(
        "--experimenter",
        default=getpass.getuser(),
        help="experimenter name (default: current user)",
    )
    args = parser.parse_args()

    videos = find_videos(VIDEO_DIR)
    print(f"Found {len(videos)} videos in {VIDEO_DIR}")

    existing = find_projects(PROJECT_DIR)
    if existing:
        print("Existing DeepLabCut projects:")
        for config in existing:
            print(f"  {config}")
        if not confirm("Create a new project anyway?"):
            raise SetupStopped("declined; kept existing project")

    try:
        import deeplabcut  # slow import; only load once we're creating a project
        from deeplabcut.utils.auxiliaryfunctions import edit_config
    except ModuleNotFoundError:
        raise SetupStopped("DeepLabCut not installed; run from the DLC environment")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    config = deeplabcut.create_new_project(
        args.name,
        args.experimenter,
        videos,
        working_directory=str(PROJECT_DIR),
        copy_videos=False,  # link videos instead of copying ~9 GB
    )
    if not config or not Path(config).exists():
        raise SetupStopped("DeepLabCut did not create a project (same name and date?)")
    edit_config(config, {"bodyparts": BODYPARTS})
    return str(config)


if __name__ == "__main__":
    try:
        config = main()
    except SetupStopped as e:
        sys.exit(f"WARNING: no project created: {e}")
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nWARNING: no project created: cancelled")
    except Exception:  # noqa: BLE001 - any crash ends in a warning
        traceback.print_exc()
        sys.exit("WARNING: no project created: unexpected error (see above)")
    print(f"SUCCESS: created DeepLabCut project: {config}")
