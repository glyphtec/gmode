"""Tests for the headless mode-image renderer (mode_images.py): colormap
anchors, frame scaling, rasterization/interpolation, edge anti-aliasing,
overlay lines and the PNG writer (read back with an independent decoder).
"""

import struct
import zlib

import numpy as np
import pytest

from soundboard_lab.mode_images import (
    JET_CONTROL_POINTS,
    OVERLAY_ORANGE,
    build_raster,
    colormap,
    make_frame,
    render_image,
    write_png,
)

# A 100 x 100 mm square as two triangles.
XY = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _pixel(frame, image, x_mm, y_mm):
    col = int((x_mm - frame.xmin) * frame.px_per_mm)
    row = int((frame.ymax - y_mm) * frame.px_per_mm)
    return image[row, col]


def _decode_png(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, header = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF, f"bad CRC in {tag!r}"
        if tag == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    width, height, depth, color_type, *_ = header
    assert (depth, color_type) == (8, 6)
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, 1 + width * 4)
    assert np.all(raw[:, 0] == 0), "writer uses filter type 0"
    return raw[:, 1:].reshape(height, width, 4)


def test_colormap_hits_the_control_points_and_clamps():
    for t, rgb in JET_CONTROL_POINTS:
        assert colormap(t) == pytest.approx(rgb)
    assert colormap(-3.0) == pytest.approx((0.0, 0.0, 1.0))
    assert colormap(9.0) == pytest.approx((1.0, 0.0, 0.0))
    assert colormap(np.array([0.125])).shape == (1, 3)
    assert colormap(0.125) == pytest.approx((0.0, 0.5, 1.0))


def test_frame_keeps_equal_scale_and_adds_a_margin():
    xy = np.array([[0.0, 0.0], [200.0, 100.0]])
    frame = make_frame(xy, width_px=400, margin_fraction=0.05)
    assert (frame.xmin, frame.xmax) == pytest.approx((-10.0, 210.0))
    assert (frame.ymin, frame.ymax) == pytest.approx((-10.0, 110.0))
    assert frame.width == 400 and frame.height == round(400 * 120 / 220)
    assert frame.px_per_mm == pytest.approx(400 / 220)


def test_linear_field_maps_to_the_expected_colors_and_the_outside_is_transparent():
    frame = make_frame(XY, width_px=200)
    raster = build_raster(frame, XY, TRIANGLES)
    image = render_image(raster, XY[:, 0] / 100.0, color_range=(0.0, 1.0))
    assert image.shape == (frame.height, frame.width, 4) and image.dtype == np.uint8

    mid = _pixel(frame, image, 50.0, 50.0)
    assert mid[0] < 12 and mid[1] == 255 and mid[2] == 0 and mid[3] == 255, f"mid value is green, got {mid}"
    left, right = _pixel(frame, image, 2.0, 50.0), _pixel(frame, image, 98.0, 50.0)
    assert left[2] > 230 and left[0] < 20, "low end is blue"
    assert right[0] > 230 and right[2] < 20, "high end is red"
    assert _pixel(frame, image, -1.0, 50.0)[3] == 0, "outside the board is transparent"
    assert image[0, 0, 3] == 0 and image[-1, -1, 3] == 0


def test_a_diagonal_boundary_is_antialiased():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    triangles = np.array([[0, 1, 2]])
    frame = make_frame(xy, width_px=150)
    image = render_image(build_raster(frame, xy, triangles), xy[:, 0], color_range=(0.0, 100.0))
    alpha = image[..., 3]
    assert ((alpha > 0) & (alpha < 255)).any(), "the silhouette must have partial-coverage pixels"
    assert (alpha == 255).sum() > 0.8 * (alpha > 0).sum()


def test_every_interior_pixel_is_covered_with_no_cracks_along_the_shared_edge():
    frame = make_frame(XY, width_px=240)
    image = render_image(build_raster(frame, XY, TRIANGLES), np.zeros(4), color_range=(-1.0, 1.0))
    inside = image[
        int((frame.ymax - 99.0) * frame.px_per_mm) : int((frame.ymax - 1.0) * frame.px_per_mm),
        int((1.0 - frame.xmin) * frame.px_per_mm) : int((99.0 - frame.xmin) * frame.px_per_mm),
    ]
    assert np.all(inside[..., 3] == 255)


def test_segments_are_drawn_orange_with_the_requested_width_in_mm():
    frame = make_frame(XY, width_px=300)
    raster = build_raster(frame, XY, TRIANGLES)
    values = np.zeros(4)
    plain = render_image(raster, values, color_range=(-1.0, 1.0))
    drawn = render_image(raster, values, color_range=(-1.0, 1.0), segments=[[10.0, 50.0, 0.0, 90.0, 50.0, 0.0]],
                         line_width_mm=4.0)

    orange = tuple(int(round(c * 255)) for c in OVERLAY_ORANGE)
    assert tuple(_pixel(frame, drawn, 50.0, 50.0)[:3]) == orange
    assert tuple(_pixel(frame, plain, 50.0, 50.0)[:3]) == (0, 255, 0)
    assert tuple(_pixel(frame, drawn, 50.0, 56.0)[:3]) == (0, 255, 0), "well away from the line stays heat-map"

    column = int((50.0 - frame.xmin) * frame.px_per_mm)
    rows = np.nonzero(np.all(drawn[:, column, :3] == orange, axis=1))[0]
    assert (rows.max() - rows.min() + 1) / frame.px_per_mm == pytest.approx(4.0, abs=2.5 / frame.px_per_mm * 2)

    # [x0, y0, x1, y1] segments are accepted too.
    four = render_image(raster, values, color_range=(-1.0, 1.0), segments=[[10.0, 50.0, 90.0, 50.0]], line_width_mm=4.0)
    assert np.array_equal(four, drawn)


def test_a_constant_field_with_a_degenerate_range_renders_the_midpoint_color():
    frame = make_frame(XY, width_px=100)
    image = render_image(build_raster(frame, XY, TRIANGLES), np.zeros(4), color_range=(0.0, 0.0))
    assert tuple(_pixel(frame, image, 50.0, 50.0)) == (0, 255, 0, 255)


def test_png_roundtrips_through_an_independent_decoder(tmp_path):
    rng = np.random.default_rng(1)
    rgba = rng.integers(0, 256, size=(37, 53, 4), dtype=np.uint8)
    path = tmp_path / "x.png"
    write_png(path, rgba)
    assert np.array_equal(_decode_png(path), rgba)
    with pytest.raises(ValueError):
        write_png(path, rgba[..., :3])
