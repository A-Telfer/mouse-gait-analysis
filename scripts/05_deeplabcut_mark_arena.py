"""Mark the 4 corners of the arena in every video of the newest DeepLabCut project.

Opens napari with one frame per video (scroll with the slider) and a 4-corner
polygon on each. Drag the corners onto the arena (press D, then drag a vertex).
Shift-C copies the current polygon to every video with the same frame size.
Corners are saved to arena_corners.json in the project when the window closes.
Run with: uv run python scripts/05_deeplabcut_mark_arena.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from _dlc_common import Stopped, latest_project, run

INSET = 0.05  # default polygon sits this far in from the frame edges


def read_frame(video: Path) -> np.ndarray:
    """Return the middle frame of a video in grayscale.

    Args:
        video: video file
    """
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise Stopped(f"could not read a frame from {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def default_corners(height: int, width: int) -> list[list[float]]:
    """Return an inset rectangle as [x, y] corners, clockwise from top-left."""
    dx, dy = width * INSET, height * INSET
    return [[dx, dy], [width - dx, dy], [width - dx, height - dy], [dx, height - dy]]


def main() -> str:
    """Open napari, then save the marked corners; return the success message."""
    config = latest_project()
    project = config.parent
    out = project / "arena_corners.json"
    print(f"Using project: {project}")

    videos = sorted((project / "videos").glob("*.mp4"))
    if not videos:
        raise Stopped(f"no videos in {project / 'videos'}")
    saved = json.loads(out.read_text()) if out.exists() else {}

    print(f"Reading a frame from {len(videos)} videos...")
    frames = [read_frame(v) for v in videos]
    sizes = [f.shape for f in frames]
    height = max(h for h, _ in sizes)
    width = max(w for _, w in sizes)
    stack = np.zeros((len(frames), height, width), dtype=np.uint8)
    for i, f in enumerate(frames):  # top-left aligned, so coords match each video
        stack[i, : f.shape[0], : f.shape[1]] = f

    def to_shape(i: int, corners: list[list[float]]) -> np.ndarray:
        return np.array([[i, y, x] for x, y in corners])

    polygons = [
        to_shape(i, saved.get(v.name) or default_corners(*sizes[i]))
        for i, v in enumerate(videos)
    ]

    import napari

    viewer = napari.Viewer(title="Mark arena corners")
    viewer.add_image(stack, name="frames")
    names = [v.name for v in videos]
    arena = viewer.add_shapes(
        polygons,
        shape_type="polygon",
        name="arena",
        edge_color="yellow",
        edge_width=3,
        face_color="transparent",
        features={"video": names},
        text={"string": "{video}", "anchor": "upper_left", "color": "yellow"},
    )
    viewer.layers.selection.active = arena

    def by_slice() -> dict[int, np.ndarray]:
        """Map slice index to its polygon; the last one drawn wins."""
        return {round(p[0, 0]): p for p in arena.data}

    @arena.bind_key("Shift-C", overwrite=True)
    def copy_to_same_size(layer: napari.layers.Shapes) -> None:
        current = int(viewer.dims.current_step[0])
        source = by_slice().get(current)
        if source is None:
            print("no polygon on this video to copy")
            return
        polys = by_slice()
        for i in range(len(videos)):
            if sizes[i] == sizes[current]:
                polys[i] = source.copy()
                polys[i][:, 0] = i
        layer.data = [polys[i] for i in sorted(polys)]
        layer.shape_type = ["polygon"] * len(layer.data)
        layer.features = {"video": [names[i] for i in sorted(polys)]}
        copied = sum(s == sizes[current] for s in sizes)
        print(f"copied polygon to {copied} videos sized {sizes[current]}")

    print(__doc__.split("Run with")[0])
    napari.run()

    corners = {}
    for i, poly in sorted(by_slice().items()):
        if len(poly) != 4:
            print(f"WARNING: {names[i]} has {len(poly)} corners, not 4; skipped")
            continue
        corners[names[i]] = [[round(x, 1), round(y, 1)] for _, y, x in poly]
    if not corners:
        raise Stopped("no 4-corner polygons to save")
    out.write_text(json.dumps(corners, indent=2) + "\n")
    missing = len(videos) - len(corners)
    note = f" ({missing} videos missing)" if missing else ""
    return f"saved corners for {len(corners)} videos to {out}{note}"


if __name__ == "__main__":
    run(main, failed="no arena corners saved")
