"""Label the 4 corners of the arena in every raw video.

Opens napari with one frame per video (scroll with the slider; the title shows
the video) and the polygon tool active. Click the 4 corners, then double-click
to finish. Shift-C copies the current polygon to every video with the same frame
size. To fix one, press D and drag its vertices, or select it and delete it.
The Arena review panel counts videos still to do (not exactly 1 shape with 4
points), has a Next button to jump to the next one, and shows a histogram of
polygon areas with a line for the current video, to spot polygons that are off.
Loads saved corners from data/arena_corners.csv, keyed by video file name. On
close or Cmd/Ctrl+S, every 4-corner polygon is written back there, replacing it.
napari's own File > Save menu writes a CSV keyed by slider position; don't use it.
Run with: uv run python scripts/00_label_arena_corners.py
"""

import csv
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from _dlc_common import (
    CORNERS_COLUMNS,
    CORNERS_CSV,
    RAW_VIDEO_DIR,
    Stopped,
    load_corners,
    run,
)


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


def order_corners(points: list[list[float]]) -> list[list[float]]:
    """Return 4 [x, y] points ordered top-left, top-right, bottom-right, bottom-left."""
    return [
        min(points, key=lambda p: p[0] + p[1]),
        max(points, key=lambda p: p[0] - p[1]),
        max(points, key=lambda p: p[0] + p[1]),
        min(points, key=lambda p: p[0] - p[1]),
    ]


def theme_colors(viewer: Any) -> dict[str, str]:
    """Return plot colors from the viewer's napari theme.

    Args:
        viewer: napari viewer
    """
    from napari.utils.theme import get_theme

    try:
        theme = get_theme(viewer.theme)
    except (KeyError, ValueError):
        theme = get_theme("dark")
    return {
        "text": theme.text.as_hex(),
        "lines": theme.highlight.as_hex(),
        "bars": theme.current.as_hex(),
        "current": theme.warning.as_hex(),
    }


def add_review_panel(
    viewer: Any,
    arena: Any,
    by_slice: Callable[[], dict[int, np.ndarray]],
    n_videos: int,
) -> None:
    """Dock a to-do count, a Next button and a histogram of polygon areas.

    A video is to do unless it has exactly 1 shape with 4 points.

    Args:
        viewer: napari viewer
        arena: shapes layer with the polygons
        by_slice: returns each video's polygon by slice index
        n_videos: number of videos (slices)
    """
    import seaborn as sns
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
    from shapely.geometry import Polygon

    panel = QWidget()
    layout = QVBoxLayout(panel)
    label = QLabel()
    next_button = QPushButton("Next to do")
    fig = Figure(figsize=(4, 3), layout="tight")
    fig.patch.set_alpha(0)  # let the panel's own background show through
    canvas = FigureCanvasQTAgg(fig)
    canvas.setStyleSheet("background: transparent")
    ax = fig.add_subplot()
    for widget in (label, next_button, canvas):
        layout.addWidget(widget)

    def to_do() -> list[int]:
        """Return slices without exactly 1 shape of 4 points."""
        counts = Counter(round(p[0, 0]) for p in arena.data)
        polygons = by_slice()
        return [i for i in range(n_videos) if counts[i] != 1 or len(polygons[i]) != 4]

    def go_next() -> None:
        remaining = to_do()
        if not remaining:
            return
        current = int(viewer.dims.current_step[0])
        later = [i for i in remaining if i > current]
        viewer.dims.set_current_step(0, (later or remaining)[0])  # wrap around

    next_button.clicked.connect(go_next)

    def redraw(event: Any = None) -> None:
        if str(getattr(event, "action", "")) in {"adding", "removing", "changing"}:
            return  # wait for the edit to finish
        remaining = to_do()
        label.setText(f"To do: {len(remaining)} of {n_videos} videos")
        next_button.setEnabled(bool(remaining))

        areas = {
            i: Polygon(p[:, [2, 1]]).area  # (x, y) from napari's (slice, y, x)
            for i, p in by_slice().items()
            if len(p) == 4
        }
        current = int(viewer.dims.current_step[0])
        colors = theme_colors(viewer)
        ax.clear()
        ax.set_facecolor("none")
        if areas:
            sns.histplot(
                list(areas.values()),
                ax=ax,
                color=colors["bars"],
                alpha=0.6,
                edgecolor=colors["lines"],
            )
        if current in areas:
            ax.axvline(areas[current], color=colors["current"], linewidth=2)
        ax.set_xlabel("arena area (px²)", color=colors["text"])
        ax.set_ylabel("Count", color=colors["text"])
        ax.set_title(
            f"{len(areas)} videos with 4 corners", fontsize=10, color=colors["text"]
        )
        ax.tick_params(colors=colors["lines"], labelcolor=colors["text"])
        for side, spine in ax.spines.items():
            spine.set_visible(side in {"left", "bottom"})
            spine.set_color(colors["lines"])
        canvas.draw_idle()

    arena.events.data.connect(redraw)
    viewer.dims.events.current_step.connect(redraw)
    viewer.events.theme.connect(redraw)
    viewer.window.add_dock_widget(panel, name="Arena review", area="right")
    redraw()


def save_corners(polygons: dict[int, np.ndarray], names: list[str]) -> str:
    """Write 4-corner polygons to the corners CSV; return the success message.

    Args:
        polygons: napari (slice, y, x) polygon by slice index
        names: video names by slice index
    """
    drawn = {}
    for i, poly in sorted(polygons.items()):
        if len(poly) != 4:
            print(f"WARNING: {names[i]} has {len(poly)} corners, not 4; skipped")
            continue
        drawn[names[i]] = order_corners([[x, y] for _, y, x in poly])
    if not drawn:
        raise Stopped(f"no 4-corner polygons; left {CORNERS_CSV} unchanged")
    with CORNERS_CSV.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(CORNERS_COLUMNS)
        for name, points in drawn.items():
            writer.writerow([name] + [round(v, 1) for p in points for v in p])
    missing = len(names) - len(drawn)
    note = f"; {missing} videos have no corners" if missing else ""
    return f"saved corners for {len(drawn)} videos to {CORNERS_CSV.name}{note}"


def take_over_save_shortcut(viewer: Any, save: Callable[[], None]) -> None:
    """Make Cmd/Ctrl+S call save instead of napari's Save Selected Layers.

    Args:
        viewer: napari viewer
        save: called on Cmd/Ctrl+S
    """
    from qtpy.QtGui import QAction, QKeySequence

    window = viewer.window._qt_window  # no public API for menu shortcuts
    shortcut = QKeySequence(QKeySequence.StandardKey.Save)
    for action in window.findChildren(QAction):
        if action.shortcut() == shortcut:
            action.setShortcut(QKeySequence())  # two owners would block both
    save_action = QAction("Save arena corners", window)
    save_action.setShortcut(shortcut)
    save_action.triggered.connect(save)
    window.addAction(save_action)


def main() -> str:
    """Open napari, then save the labeled corners; return the success message."""
    videos = sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise Stopped(f"no videos in {RAW_VIDEO_DIR}")
    print(f"Reading a frame from {len(videos)} videos...")
    frames = [read_frame(v) for v in videos]
    sizes = [f.shape for f in frames]
    height = max(h for h, _ in sizes)
    width = max(w for _, w in sizes)
    stack = np.zeros((len(frames), height, width), dtype=np.uint8)
    for i, f in enumerate(frames):  # top-left aligned, so coords match each video
        stack[i, : f.shape[0], : f.shape[1]] = f

    names = [v.name for v in videos]
    saved = load_corners(CORNERS_CSV)
    polygons = [
        np.array([[i, y, x] for x, y in saved[n]])
        for i, n in enumerate(names)
        if saved.get(n)
    ]
    print(f"Loaded corners for {len(polygons)} videos from {CORNERS_CSV}")

    import napari

    viewer = napari.Viewer()
    viewer.add_image(stack, name="frames")
    arena = viewer.add_shapes(
        polygons or None,
        ndim=3,
        shape_type="polygon",
        name="arena",
        edge_color="yellow",
        edge_width=3,
        face_color="transparent",
    )
    viewer.layers.selection.active = arena
    arena.mode = "add_polygon"

    def show_video(_: object = None) -> None:
        i = int(viewer.dims.current_step[0])
        viewer.title = f"Arena corners: {names[i]} ({i + 1}/{len(names)})"

    viewer.dims.events.current_step.connect(show_video)
    show_video()

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
        copied = sum(s == sizes[current] for s in sizes)
        print(f"copied polygon to {copied} videos sized {sizes[current]}")

    add_review_panel(viewer, arena, by_slice, len(videos))

    def save() -> None:
        try:
            message = save_corners(by_slice(), names)
        except Stopped as e:
            message = f"not saved: {e}"
        print(message)
        viewer.status = message

    take_over_save_shortcut(viewer, save)
    print(__doc__.split("Run with")[0])
    napari.run()

    return save_corners(by_slice(), names)


if __name__ == "__main__":
    run(main, failed="no arena corners saved")
