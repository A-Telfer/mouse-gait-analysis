"""Create a DeepLabCut project in data/deeplabcut from the videos in data/raw-videos.

Run with the Python environment that has DeepLabCut installed.
"""

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "raw-videos"
PROJECT_DIR = ROOT / "data" / "deeplabcut"


def find_projects(project_dir: Path) -> list[Path]:
    """Return config.yaml paths of existing projects.

    Args:
        project_dir: folder holding DLC projects
    """
    return sorted(project_dir.glob("*/config.yaml"))


def confirm(prompt: str) -> bool:
    """Ask a yes/no question; default no."""
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="mouse-gait", help="project name")
    parser.add_argument(
        "--experimenter", default=getpass.getuser(), help="experimenter name"
    )
    args = parser.parse_args()

    if not VIDEO_DIR.is_dir():
        sys.exit(f"Video folder not found: {VIDEO_DIR}")
    paths = sorted(VIDEO_DIR.glob("*.mp4"))
    missing = [p for p in paths if not p.is_file()]  # e.g. broken links
    if missing:
        sys.exit("Unreadable videos:\n" + "\n".join(f"  {p}" for p in missing))
    if not paths:
        sys.exit(f"No .mp4 videos found in {VIDEO_DIR}")
    videos = [str(p) for p in paths]
    print(f"Found {len(videos)} videos in {VIDEO_DIR}")

    existing = find_projects(PROJECT_DIR)
    if existing:
        print("Existing DeepLabCut projects:")
        for config in existing:
            print(f"  {config}")
        if not confirm("Create a new project anyway?"):
            print("No project created.")
            return

    import deeplabcut  # slow import; only load once we're creating a project

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    config = deeplabcut.create_new_project(
        args.name,
        args.experimenter,
        videos,
        working_directory=str(PROJECT_DIR),
        copy_videos=False,  # link videos instead of copying ~9 GB
    )
    if not config or not Path(config).exists():
        sys.exit("DeepLabCut did not create a project (same name and date exists?)")
    print(f"Created project with {len(videos)} videos: {config}")


if __name__ == "__main__":
    main()
