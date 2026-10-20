"""Extract the initial frames to label from every video in the DeepLabCut project.

Uses DLC's k-means extraction, which picks visually different frames from each
video. Run with the Python environment that has DeepLabCut installed.
"""

import argparse
from pathlib import Path

from _dlc_common import Stopped, choose_project, confirm, import_deeplabcut, run


def count_frames(project: Path) -> int:
    """Count extracted frames already in labeled-data.

    Args:
        project: DLC project folder
    """
    return len(list((project / "labeled-data").glob("*/*.png")))


def main() -> str:
    """Extract frames; return the success message."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=5,
        help="frames to extract per video (default: 5)",
    )
    parser.add_argument(
        "--algo",
        choices=["kmeans", "uniform"],
        default="kmeans",
        help="frame selection method (default: kmeans)",
    )
    parser.add_argument(
        "--cluster-step",
        type=int,
        default=10,
        help="kmeans: use every Nth frame, for speed (default: 10)",
    )
    args = parser.parse_args()
    if args.frames_per_video < 1:
        raise Stopped("--frames-per-video must be at least 1")

    config = choose_project()
    project = config.parent
    print(f"Using project: {project}")

    before = count_frames(project)
    if before and not confirm(f"{before} frames already extracted. Add more?"):
        raise Stopped("declined; kept existing frames")

    deeplabcut = import_deeplabcut()
    from deeplabcut.utils.auxiliaryfunctions import edit_config

    edit_config(str(config), {"numframes2pick": args.frames_per_video})
    deeplabcut.extract_frames(
        str(config),
        mode="automatic",
        algo=args.algo,
        userfeedback=False,  # extract from every video without asking per video
        cluster_step=args.cluster_step,
        crop=False,
    )

    added = count_frames(project) - before
    if added <= 0:
        raise Stopped("DeepLabCut extracted no frames (see output above)")
    return f"extracted {added} frames to {project / 'labeled-data'}"


if __name__ == "__main__":
    run(main, failed="no frames extracted")
