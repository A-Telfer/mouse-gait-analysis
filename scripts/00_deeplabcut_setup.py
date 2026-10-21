"""Create a DeepLabCut project in data/deeplabcut from the videos in data/videos/cropped.

Run with: uv run python scripts/02_deeplabcut_setup.py
"""

import argparse
import getpass
from pathlib import Path

from _dlc_common import (
    CROPPED_VIDEO_DIR,
    PROJECT_DIR,
    Stopped,
    confirm,
    find_projects,
    import_deeplabcut,
    run,
)

BODYPARTS = [
    "nose",
    "chest",
    "mid_back",
    "tailbase",
    "midtail",
    "tail_tip",
    "left_front_paw",
    "right_front_paw",
    "left_front_toes",
    "right_front_toes",
    "left_toes",
    "left_back_paw",
    "right_toes",
    "right_back_paw",
]


def find_videos(video_dir: Path) -> list[str]:
    """Return paths of all .mp4 videos; stop if any are missing.

    Args:
        video_dir: folder of videos
    """
    if not video_dir.is_dir():
        raise Stopped(f"{video_dir} not found; run 01_crop_videos.py first")
    paths = sorted(video_dir.glob("*.mp4"))
    missing = [p for p in paths if not p.is_file()]  # e.g. broken links
    if missing:
        listing = "\n".join(f"  {p}" for p in missing)
        raise Stopped(f"unreadable videos:\n{listing}")
    if not paths:
        raise Stopped(f"no .mp4 videos found in {video_dir}")
    return [str(p) for p in paths]


def main() -> str:
    """Create the project; return the success message."""
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

    videos = find_videos(CROPPED_VIDEO_DIR)
    print(f"Found {len(videos)} videos in {CROPPED_VIDEO_DIR}")

    existing = find_projects()
    if existing:
        print("Existing DeepLabCut projects:")
        for config in existing:
            print(f"  {config}")
        if not confirm("Create a new project anyway?"):
            raise Stopped("declined; kept existing project")

    deeplabcut = import_deeplabcut()
    from deeplabcut.utils.auxiliaryfunctions import edit_config

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    config = deeplabcut.create_new_project(
        args.name,
        args.experimenter,
        videos,
        working_directory=str(PROJECT_DIR),
        copy_videos=False,  # link videos instead of copying ~9 GB
    )
    if not config or not Path(config).exists():
        raise Stopped("DeepLabCut did not create a project (same name and date?)")
    edit_config(config, {"bodyparts": BODYPARTS})
    return f"created DeepLabCut project: {config}"


if __name__ == "__main__":
    run(main, failed="no project created")
