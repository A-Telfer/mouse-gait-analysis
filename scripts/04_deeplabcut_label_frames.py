"""Open the newest DeepLabCut project in the labeling GUI to label extracted frames.

Run with: uv run python scripts/04_deeplabcut_label_frames.py
"""

from _dlc_common import Stopped, import_deeplabcut, latest_project, run


def main() -> str:
    """Open the labeling GUI; return the success message."""
    config = latest_project()
    project = config.parent
    print(f"Using project: {project}")

    if not any((project / "labeled-data").glob("*/*.png")):
        raise Stopped(
            "no extracted frames; run 03_deeplabcut_extract_uniform_frames.py first"
        )

    deeplabcut = import_deeplabcut()
    import napari

    deeplabcut.label_frames(str(config))
    napari.run()  # label_frames only builds the window; this keeps it open
    return f"labeling GUI closed for {project.name}"


if __name__ == "__main__":
    run(main, failed="labeling not opened")
