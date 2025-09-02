"""
Tests for GEOSClipByRect based on unit tests from libgeos.

There are some expected differences due to Shapely's handling of empty
geometries.
"""

import pytest

from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt, loads as load_wkt


def test_point_outside():
    """Point outside (augmented for larger and negative coordinates)"""
    geom1 = load_wkt("POINT (-1000.5 99999.5)")
    geom2 = clip_by_rect(geom1, 0, 0, 500, 500)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def test_point_inside():
    """Point inside (augmented for a large floating point)"""
    geom1 = load_wkt("POINT (250.9999 123.0001)")
    geom2 = clip_by_rect(geom1, 200, 100, 300, 200)
    assert dump_wkt(geom2, rounding_precision=0) == "POINT (251 123)"


def test_point_on_boundary():
    """Point on boundary (augmented: lower-left corner of rect boundary with negative numbers)"""
    geom1 = load_wkt("POINT (-10 -10)")
    geom2 = clip_by_rect(geom1, -10, -10, 10, 10)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def test_line_outside():
    """Line outside (augmented: vertical line completely out on the right, negative/positive)"""
    geom1 = load_wkt("LINESTRING (1000 1000, 2000 3000)")
    geom2 = clip_by_rect(geom1, -100, -100, 100, 100)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def test_line_inside():
    """Line inside (augmented: float/negative coordinates inside the rectangle)"""
    geom1 = load_wkt("LINESTRING (-1.1 -2.2, 0.3 -0.4)")
    geom2 = clip_by_rect(geom1, -2, -3, 1, 1)
    assert dump_wkt(geom2, rounding_precision=0) == "LINESTRING (-1 -2, 0 -0)"


def test_line_on_boundary():
    """Line on boundary (augmented: horizontal along max y boundary)"""
    geom1 = load_wkt("LINESTRING (0 50, 10 50, 20 50)")
    geom2 = clip_by_rect(geom1, 0, 0, 20, 50)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def test_line_splitting_rectangle():
    """Line splitting rectangle (augmented: line starts before and ends after rectangle, partly inside)"""
    geom1 = load_wkt("LINESTRING (-5 -5, 10 10, 25 25)")
    geom2 = clip_by_rect(geom1, 0, 0, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "LINESTRING (0 0, 10 10, 20 20)"


@pytest.mark.xfail(reason="TODO issue to CCW")
def test_polygon_shell_ccw_fully_on_rectangle_boundary():
    """Polygon shell (CCW) fully on rectangle boundary, augmented with larger square"""
    geom1 = load_wkt("POLYGON ((-50 -50, 100 -50, 100 100, -50 100, -50 -50))")
    geom2 = clip_by_rect(geom1, -50, -50, 100, 100)
    assert (
        dump_wkt(geom2, rounding_precision=0)
        == "POLYGON ((-50 -50, 100 -50, 100 100, -50 100, -50 -50))"
    )


@pytest.mark.xfail(reason="TODO issue to CW")
def test_polygon_shell_cc_fully_on_rectangle_boundary():
    """Polygon shell (CW) fully on rectangle boundary, augmented with large square, reversed order"""
    geom1 = load_wkt("POLYGON ((-50 -50, -50 100, 100 100, 100 -50, -50 -50))")
    geom2 = clip_by_rect(geom1, -50, -50, 100, 100)
    assert (
        dump_wkt(geom2, rounding_precision=0)
        == "POLYGON ((-50 -50, 100 -50, 100 100, -50 100, -50 -50))"
    )


def polygon_hole_ccw_fully_on_rectangle_boundary():
    """Polygon hole (CCW) fully on rectangle boundary"""
    geom1 = load_wkt(
        "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))"
    )
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def polygon_hole_cw_fully_on_rectangle_boundary():
    """Polygon hole (CW) fully on rectangle boundary"""
    geom1 = load_wkt(
        "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 10 20, 20 20, 20 10, 10 10))"
    )
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def polygon_fully_within_rectangle():
    """Polygon fully within rectangle"""
    wkt = "POLYGON ((1 1, 1 30, 30 30, 30 1, 1 1), (10 10, 20 10, 20 20, 10 20, 10 10))"
    geom1 = load_wkt(wkt)
    geom2 = clip_by_rect(geom1, 0, 0, 40, 40)
    assert dump_wkt(geom2, rounding_precision=0) == wkt


def polygon_overlapping_rectangle():
    """Polygon overlapping rectangle"""
    wkt = "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))"
    geom1 = load_wkt(wkt)
    geom2 = clip_by_rect(geom1, 5, 5, 15, 15)
    assert (
        dump_wkt(geom2, rounding_precision=0)
        == "POLYGON ((5 5, 5 15, 10 15, 10 10, 15 10, 15 5, 5 5))"
    )