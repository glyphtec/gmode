"""Static, top-down heat-map images of a mode on the solved soundboard
footprint -- the displacement and strain-energy views of the Blender
Results panel, rendered without Blender (or any GL context, camera or
font) so an experiment export is deterministic, runs headless in the
engine venv, and can be re-rendered later from the exported fields.

Pure numpy: the footprint's triangles are rasterized once into a
`Raster` (covered pixels, owning triangle, barycentric weights) that is
reused for every mode, a mode's per-vertex values are interpolated
through it, mapped through the colormap and overlaid with anti-aliased
line segments, and the result is box-downsampled from a supersampled
buffer and written as an RGBA PNG (transparent outside the board).
World +X is image right, +Y is image up -- the same orientation as the
viewport seen from above.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The same blue -> cyan -> green -> yellow -> red ramp as the add-on's
# `_jet_colormap` (addon/soundboard_lab/analysis.py): the two cannot
# share code (the add-on never imports the engine), so keep them equal.
JET_CONTROL_POINTS: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (0.00, (0.0, 0.0, 1.0)),
    (0.25, (0.0, 1.0, 1.0)),
    (0.50, (0.0, 1.0, 0.0)),
    (0.75, (1.0, 1.0, 0.0)),
    (1.00, (1.0, 0.0, 0.0)),
)

# Contour lines and bending ticks, matching the add-on's overlay orange.
OVERLAY_ORANGE = (1.0, 0.45, 0.0)


def colormap(t: np.ndarray) -> np.ndarray:
    """(..., 3) float RGB in [0, 1] for `t` clamped to [0, 1]."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    xs = [p[0] for p in JET_CONTROL_POINTS]
    return np.stack(
        [np.interp(t, xs, [p[1][channel] for p in JET_CONTROL_POINTS]) for channel in range(3)], axis=-1
    )


@dataclass(frozen=True)
class ImageFrame:
    """The rectangle of the footprint plane an image covers, in mm, and
    its size in pixels (the same scale in x and y)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int

    @property
    def px_per_mm(self) -> float:
        return self.width / (self.xmax - self.xmin)


def make_frame(xy: np.ndarray, width_px: int, margin_fraction: float = 0.03) -> ImageFrame:
    """A frame around `xy` with a margin of `margin_fraction` of the
    larger extent, `width_px` wide, and the height that keeps the scale
    equal in both axes."""
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    margin = margin_fraction * float((hi - lo).max())
    xmin, ymin = lo - margin
    xmax, ymax = hi + margin
    height = max(1, int(round(width_px * (ymax - ymin) / (xmax - xmin))))
    return ImageFrame(float(xmin), float(ymin), float(xmax), float(ymax), int(width_px), height)


@dataclass(frozen=True)
class Raster:
    """Triangle coverage of a frame at `supersample` times its pixel
    resolution: for each covered high-resolution pixel, its flat index,
    owning triangle and barycentric weights. Geometry-only, so one
    `Raster` serves every mode of an experiment."""

    frame: ImageFrame
    supersample: int
    triangles: np.ndarray  # (T, 3) vertex indices
    pixel: np.ndarray  # (N,) flat index into the (H*ss, W*ss) buffer
    triangle: np.ndarray  # (N,) owning triangle
    weights: np.ndarray  # (N, 3) barycentric weights

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame.height * self.supersample, self.frame.width * self.supersample

    def segments_to_pixels(self, segments) -> np.ndarray:
        """(S, 4) high-resolution pixel `[x0, y0, x1, y1]` for footprint
        segments given either that way (mm) or as the engine's
        `[x0, y0, z0, x1, y1, z1]`."""
        seg = np.asarray(segments, dtype=np.float64)
        if seg.ndim != 2 or seg.shape[1] not in (4, 6):
            raise ValueError("segments must be (S, 4) or (S, 6)")
        if seg.shape[1] == 6:
            seg = seg[:, [0, 1, 3, 4]]
        scale = self.frame.px_per_mm * self.supersample
        return np.column_stack(
            [
                (seg[:, 0] - self.frame.xmin) * scale,
                (self.frame.ymax - seg[:, 1]) * scale,
                (seg[:, 2] - self.frame.xmin) * scale,
                (self.frame.ymax - seg[:, 3]) * scale,
            ]
        )


def build_raster(frame: ImageFrame, xy: np.ndarray, triangles: np.ndarray, supersample: int = 2) -> Raster:
    height, width = frame.height * supersample, frame.width * supersample
    scale = frame.px_per_mm * supersample
    px = (xy[:, 0] - frame.xmin) * scale
    py = (frame.ymax - xy[:, 1]) * scale

    pixel_parts, triangle_parts, weight_parts = [], [], []
    for t, (ia, ib, ic) in enumerate(triangles):
        ax, ay, bx, by, cx, cy = px[ia], py[ia], px[ib], py[ib], px[ic], py[ic]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        j0 = max(int(np.ceil(min(ax, bx, cx) - 0.5)), 0)
        j1 = min(int(np.floor(max(ax, bx, cx) - 0.5)), width - 1)
        i0 = max(int(np.ceil(min(ay, by, cy) - 0.5)), 0)
        i1 = min(int(np.floor(max(ay, by, cy) - 0.5)), height - 1)
        if j0 > j1 or i0 > i1:
            continue
        jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
        x = jj + 0.5
        y = ii + 0.5
        l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det
        l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if not inside.any():
            continue
        pixel_parts.append((ii[inside] * width + jj[inside]).astype(np.int64))
        triangle_parts.append(np.full(int(inside.sum()), t, dtype=np.int32))
        weight_parts.append(np.column_stack([l1[inside], l2[inside], l3[inside]]).astype(np.float32))

    if not pixel_parts:
        raise ValueError("the footprint covers no pixels")
    pixel = np.concatenate(pixel_parts)
    # A pixel on a shared edge is claimed by two triangles; keep one.
    _, first = np.unique(pixel, return_index=True)
    return Raster(
        frame=frame,
        supersample=supersample,
        triangles=np.asarray(triangles),
        pixel=pixel[first],
        triangle=np.concatenate(triangle_parts)[first],
        weights=np.concatenate(weight_parts)[first],
    )


def _draw_segments(rgba: np.ndarray, segments_px: np.ndarray, half_width_px: float, color: tuple) -> None:
    """Opaque round-capped thick lines, in place on a high-resolution
    (H, W, 4) buffer; anti-aliasing comes from the later downsample."""
    height, width = rgba.shape[:2]
    pad = half_width_px + 1.0
    paint = np.array([*color, 1.0], dtype=rgba.dtype)
    for x0, y0, x1, y1 in segments_px:
        j0 = max(int(np.floor(min(x0, x1) - pad)), 0)
        j1 = min(int(np.ceil(max(x0, x1) + pad)), width - 1)
        i0 = max(int(np.floor(min(y0, y1) - pad)), 0)
        i1 = min(int(np.ceil(max(y0, y1) + pad)), height - 1)
        if j0 > j1 or i0 > i1:
            continue
        jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
        x = jj + 0.5
        y = ii + 0.5
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / length_sq, 0.0, 1.0) if length_sq > 0.0 else 0.0
        distance = np.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
        block = rgba[i0 : i1 + 1, j0 : j1 + 1]
        block[distance <= half_width_px] = paint


def render_image(
    raster: Raster,
    values: np.ndarray,
    color_range: tuple[float, float],
    segments: list | None = None,
    line_width_mm: float = 0.6,
    line_color: tuple[float, float, float] = OVERLAY_ORANGE,
) -> np.ndarray:
    """(H, W, 4) uint8 RGBA of the per-vertex `values`, colored over
    `color_range` (low, high) and transparent outside the footprint,
    with `segments` (contours, ticks; see `Raster.segments_to_pixels`)
    drawn on top."""
    low, high = color_range
    height, width = raster.shape
    ss = raster.supersample
    rgba = np.zeros((height * width, 4), dtype=np.float32)

    per_pixel = np.einsum("nk,nk->n", raster.weights, values[raster.triangles[raster.triangle]].astype(np.float32))
    t = (per_pixel - low) / (high - low) if high > low else np.full_like(per_pixel, 0.5)
    rgba[raster.pixel, :3] = colormap(t)
    rgba[raster.pixel, 3] = 1.0
    rgba = rgba.reshape(height, width, 4)

    if segments is not None and len(segments):
        half_width_px = 0.5 * line_width_mm * raster.frame.px_per_mm * ss
        _draw_segments(rgba, raster.segments_to_pixels(segments), half_width_px, line_color)

    # Premultiplied box downsample, so the board's edge fades correctly.
    frame = raster.frame
    premultiplied = rgba.copy()
    premultiplied[..., :3] *= premultiplied[..., 3:4]
    small = premultiplied.reshape(frame.height, ss, frame.width, ss, 4).mean(axis=(1, 3))
    alpha = small[..., 3:4]
    rgb = np.divide(small[..., :3], alpha, out=np.zeros_like(small[..., :3]), where=alpha > 0)
    out = np.concatenate([rgb, alpha], axis=-1)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def write_png(path: Path, rgba: np.ndarray) -> None:
    """Write an (H, W, 4) uint8 array as an 8-bit RGBA PNG."""
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("write_png needs an (H, W, 4) uint8 array")
    height, width, _ = rgba.shape
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rgba.reshape(height, width * 4)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
        + chunk(b"IEND", b"")
    )
    Path(path).write_bytes(png)
