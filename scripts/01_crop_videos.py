"""Crop every raw video to the arena, using the corners from 00_label_arena_corners.py.

Reads data/arena_corners.csv and the videos in data/videos/raw, and writes cropped
videos to data/videos/cropped, all the same size. Nothing is re-encoded: ffmpeg's
h264_metadata filter stores the crop as a cropping window in the H.264 header,
which every decoder applies, so cropping is fast and lossless.

The size fits the largest arena plus a 10 px margin on each side, but no more than
the smallest video. Each box is centred on its arena vertically. A box's left edge
must be a multiple of 64 px, since OpenCV (and so DeepLabCut) rounds a left crop
down to that and would otherwise read shifted frames; so horizontally each box takes
the allowed position that clips the least arena. Videos whose arena is clipped are
listed. Each video's box, in raw-video pixels, is written to crop_boxes.csv next to
the cropped videos, and the corners shifted into cropped-video pixels (minus the
box's x, y) to data/arena_corners_cropped.csv. A
cropped video is skipped if it exists and its box hasn't changed.
Run with: uv run python scripts/01_crop_videos.py
"""

import csv
import math
import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import cv2
import imageio_ffmpeg
from _dlc_common import (
    CORNERS_COLUMNS,
    CORNERS_CSV,
    CROPPED_VIDEO_DIR,
    RAW_VIDEO_DIR,
    Stopped,
    load_corners,
    run,
)
from tqdm import tqdm

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # bundled binary; no system ffmpeg needed
BOXES_CSV = CROPPED_VIDEO_DIR / "crop_boxes.csv"
CROPPED_CORNERS_CSV = CORNERS_CSV.with_name("arena_corners_cropped.csv")
LEFT_ALIGN = 64  # px; OpenCV rounds a left crop down to a multiple of this
MARGIN = 10  # px of floor kept around the arena


class Header(NamedTuple):
    """Frame geometry from a video's H.264 header, in pixels."""

    coded_width: int
    coded_height: int
    crop_left: int  # existing crop: where the displayed frame starts
    crop_top: int
    width: int  # displayed size
    height: int


class Box(NamedTuple):
    """Crop rectangle in displayed raw-video pixels."""

    x: int
    y: int
    width: int
    height: int


def in_order(points: list[list[float]]) -> bool:
    """Check corners run top-left, top-right, bottom-right, bottom-left.

    Args:
        points: 4 [x, y] corners; y grows downwards, as in image pixels
    """
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    left_of_right = x0 < x1 and x0 < x2 and x3 < x1 and x3 < x2
    top_above_bottom = y0 < y2 and y0 < y3 and y1 < y2 and y1 < y3
    return left_of_right and top_above_bottom


def read_header(video: Path) -> Header:
    """Read frame size and existing crop from the first H.264 SPS.

    Args:
        video: video file
    """
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(video), "-map", "0:v:0", "-frames:v", "1"]
        + ["-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    if not re.search(r"Video: h264\b", result.stderr):
        raise Stopped(
            f"{video.name} is not H.264, so can't be cropped without re-encoding"
        )
    fields: dict[str, int] = {}
    for name, value in re.findall(r"\s(\w+)\s+[01]+ = (\d+)", result.stderr):
        fields.setdefault(name, int(value))  # first SPS only
    if fields.get("chroma_format_idc") != 1 or fields.get("frame_mbs_only_flag") != 1:
        raise Stopped(f"{video.name}: only progressive 4:2:0 H.264 is supported")
    crop = {  # stored in 2 px units for 4:2:0
        side: 2 * fields.get(f"frame_crop_{side}_offset", 0)
        for side in ("left", "right", "top", "bottom")
    }
    if crop["left"] % LEFT_ALIGN:
        raise Stopped(f"{video.name} already has an unaligned left crop")
    coded_width = 16 * (fields["pic_width_in_mbs_minus1"] + 1)
    coded_height = 16 * (fields["pic_height_in_map_units_minus1"] + 1)
    return Header(
        coded_width,
        coded_height,
        crop["left"],
        crop["top"],
        coded_width - crop["left"] - crop["right"],
        coded_height - crop["top"] - crop["bottom"],
    )


class Extent(NamedTuple):
    """Arena bounding box in displayed pixels, clipped to the frame."""

    x0: float
    y0: float
    x1: float
    y1: float


def arena_extent(points: list[list[float]], header: Header) -> Extent:
    """Return the corners' bounding box, clipped to the frame.

    Args:
        points: 4 [x, y] corners in displayed pixels
        header: the video's frame geometry
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return Extent(
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(float(header.width), max(xs)),
        min(float(header.height), max(ys)),
    )


def even_up(v: float) -> int:
    """Round up to an even integer."""
    return math.ceil(v / 2) * 2


def common_size(extents: list[Extent], headers: list[Header]) -> tuple[int, int]:
    """Return one even (width, height) that fits every arena plus margin.

    The width allows for each box's left edge snapping down to LEFT_ALIGN. Both
    are capped at the smallest frame, since a crop can't add pixels.

    Args:
        extents: arena extents
        headers: matching frame geometries
    """
    widths, heights = [], []
    for e, h in zip(extents, headers):
        left = max(0, math.floor(e.x0 - MARGIN) // LEFT_ALIGN * LEFT_ALIGN)
        widths.append(min(h.width, e.x1 + MARGIN) - left)
        heights.append(min(h.height, e.y1 + MARGIN) - max(0, e.y0 - MARGIN))
    max_width = min(h.width for h in headers) // 2 * 2
    max_height = min(h.height for h in headers) // 2 * 2
    return min(max_width, even_up(max(widths))), min(max_height, even_up(max(heights)))


def place_box(extent: Extent, header: Header, width: int, height: int) -> Box:
    """Return a width x height box around the arena, aligned and inside the frame.

    Args:
        extent: arena extent
        header: the video's frame geometry
        width: box width
        height: box height
    """
    centre_x = (extent.x0 + extent.x1) / 2

    def badness(left: int) -> tuple[float, float]:
        """Arena px clipped, then distance from centred."""
        clipped = max(0, left - extent.x0) + max(0, extent.x1 - left - width)
        return clipped, abs(left + width / 2 - centre_x)

    left = min(range(0, header.width - width + 1, LEFT_ALIGN), key=badness)
    top = round((extent.y0 + extent.y1 - height) / 4) * 2  # centred, even
    top = max(0, min(top, (header.height - height) // 2 * 2))
    return Box(left, top, width, height)


def margin(extent: Extent, box: Box) -> float:
    """Return the smallest gap between arena and box edge; negative if clipped.

    Args:
        extent: arena extent
        box: crop box
    """
    return min(
        extent.x0 - box.x,
        extent.y0 - box.y,
        box.x + box.width - extent.x1,
        box.y + box.height - extent.y1,
    )


def crop(src: Path, dst: Path, box: Box, header: Header) -> None:
    """Write src cropped to box, copying all streams without re-encoding.

    Args:
        src: raw video
        dst: cropped video to write
        box: crop rectangle in displayed pixels
        header: src's frame geometry
    """
    left = header.crop_left + box.x
    top = header.crop_top + box.y
    right = header.coded_width - left - box.width
    bottom = header.coded_height - top - box.height
    bsf = (
        f"h264_metadata=crop_left={left}:crop_right={right}"
        f":crop_top={top}:crop_bottom={bottom}"
    )
    part = dst.with_suffix(".part.mp4")  # an interrupted run leaves no bad video
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
        + ["-map", "0", "-c", "copy", "-bsf:v", bsf, str(part)],
        check=True,
    )
    cap = cv2.VideoCapture(str(part))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if (width, height) != (box.width, box.height):
        part.unlink()
        raise Stopped(
            f"{dst.name}: OpenCV reads {width}x{height}, "
            f"expected {box.width}x{box.height}"
        )
    part.replace(dst)


def usable_corners(videos: list[Path]) -> dict[str, list[list[float]]]:
    """Return 4 in-order corners by video name; warn about the rest.

    Args:
        videos: raw videos
    """
    corners = load_corners(CORNERS_CSV)
    names = {v.name for v in videos}
    unknown = sorted(corners.keys() - names)
    if unknown:  # e.g. a video renamed or removed after its corners were marked
        print(f"Corners for videos not in {RAW_VIDEO_DIR} (ignored):")
        print("\n".join(f"  {n}" for n in unknown))
    missing = sorted(names - corners.keys())
    if missing:
        print(
            f"WARNING: {len(missing)} of {len(videos)} videos have no corners and "
            f"will be skipped; mark them with 00_label_arena_corners.py:"
        )
        print("\n".join(f"  {n}" for n in missing))

    marked = names & corners.keys()
    incomplete = sorted(n for n in marked if len(corners[n]) != 4)
    if incomplete:
        print(
            f"WARNING: {len(incomplete)} videos don't have exactly 4 corners and "
            f"will be skipped; re-mark them with 00_label_arena_corners.py:"
        )
        print("\n".join(f"  {n}: {len(corners[n])} points" for n in incomplete))

    complete = marked - set(incomplete)
    bad = sorted(n for n in complete if not in_order(corners[n]))
    if bad:
        print(
            f"WARNING: {len(bad)} videos have corners out of order (expected "
            f"top-left, top-right, bottom-right, bottom-left) and will be skipped; "
            f"re-mark them with 00_label_arena_corners.py:"
        )
        print("\n".join(f"  {n}: {corners[n]}" for n in bad))
    return {n: corners[n] for n in sorted(complete - set(bad))}


def load_boxes() -> dict[str, list[str]]:
    """Return saved crop box rows by video name, or {} if there's no file."""
    if not BOXES_CSV.exists():
        return {}
    with BOXES_CSV.open() as f:
        return {row[0]: row for row in list(csv.reader(f))[1:] if row}


def save_boxes(boxes: dict[str, list[str]]) -> None:
    """Write crop box rows to the boxes CSV, sorted by video name.

    Args:
        boxes: [video, x, y, width, height] rows by video name
    """
    with BOXES_CSV.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["video", "x", "y", "width", "height"])
        writer.writerows(boxes[n] for n in sorted(boxes))


def save_cropped_corners(
    corners: dict[str, list[list[float]]], boxes: dict[str, list[str]]
) -> None:
    """Write corners in cropped-video pixels for every cropped video.

    Corners of a clipped arena can fall outside the cropped frame.

    Args:
        corners: raw-video [x, y] corners by video name
        boxes: [video, x, y, width, height] rows by video name
    """
    with CROPPED_CORNERS_CSV.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(CORNERS_COLUMNS)
        for name in sorted(corners.keys() & boxes.keys()):
            dx, dy = int(boxes[name][1]), int(boxes[name][2])
            shifted = [[x - dx, y - dy] for x, y in corners[name]]
            writer.writerow([name] + [round(v, 1) for p in shifted for v in p])


def main() -> str:
    """Crop the videos; return the success message."""
    videos = sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise Stopped(f"no videos in {RAW_VIDEO_DIR}")
    if not CORNERS_CSV.exists():
        raise Stopped(f"{CORNERS_CSV} not found; run 00_label_arena_corners.py first")
    corners = usable_corners(videos)
    if not corners:
        raise Stopped("no videos with usable corners")

    headers = {
        n: read_header(RAW_VIDEO_DIR / n)
        for n in tqdm(corners, desc="Reading video headers", unit="video")
    }
    extents = {n: arena_extent(corners[n], headers[n]) for n in corners}
    width, height = common_size(list(extents.values()), list(headers.values()))
    print(f"Crop size for all videos: {width}x{height}")
    planned = {n: place_box(extents[n], headers[n], width, height) for n in corners}

    margins = {n: margin(extents[n], planned[n]) for n in corners}
    clipped = sorted(n for n, m in margins.items() if m < 0)
    if clipped:
        print(
            f"WARNING: {len(clipped)} videos have arena outside the box, as the box "
            f"can only start at multiples of {LEFT_ALIGN} px:"
        )
        print("\n".join(f"  {n}: {-margins[n]:.0f} px clipped" for n in clipped))
    tight = sum(0 <= m < MARGIN for m in margins.values())
    if tight:
        print(
            f"{tight} more videos have under {MARGIN} px of margin, mostly arenas "
            f"reaching the frame edge."
        )

    CROPPED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    boxes = load_boxes()
    rows = {n: [n, *map(str, planned[n])] for n in corners}
    todo = [
        n
        for n in corners
        if boxes.get(n) != rows[n] or not (CROPPED_VIDEO_DIR / n).exists()
    ]
    skipped = len(corners) - len(todo)
    try:
        progress = tqdm(todo, desc="Cropping", unit="video")
        for name in progress:
            progress.set_postfix_str(name)
            crop(
                RAW_VIDEO_DIR / name,
                CROPPED_VIDEO_DIR / name,
                planned[name],
                headers[name],
            )
            boxes[name] = rows[name]
    finally:  # keep the boxes of videos cropped before an error or Ctrl-C
        save_boxes(boxes)
        save_cropped_corners(corners, boxes)

    note = f"; {skipped} were already cropped to the same box" if skipped else ""
    return (
        f"cropped {len(todo)} videos to {width}x{height} into {CROPPED_VIDEO_DIR}{note}"
    )


if __name__ == "__main__":
    run(main, failed="cropping stopped")
