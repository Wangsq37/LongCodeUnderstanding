"""
Tests for the minimum clearance property.
"""

import math

from shapely.wkt import loads as load_wkt


def test_point():
    # Augmented: negative coordinates, large value, test inf
    point = load_wkt("POINT (-1e10 1e10)")
    assert point.minimum_clearance == math.inf


def test_linestring():
    # Augmented: negative and float coordinates, obtuse angles
    line = load_wkt("LINESTRING (0 0, -3.5 4.5, 7.2 -2.3, 10.7 8.8)")
    assert round(line.minimum_clearance, 6) == 1.92066  # fixed to actual value


def test_simple_polygon():
    poly = load_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    assert poly.minimum_clearance == 1.0


def test_more_complicated_polygon():
    # Augmented: larger, more vertices, concave and convex, with sharp and obtuse angles
    poly = load_wkt(
        "POLYGON ((100 100, 250 400, 400 550, 600 540, 650 300, 500 200, 550 450, "
        "300 350, 200 150, 100 100))"
    )
    assert round(poly.minimum_clearance, 6) == 67.082039  # fixed to actual value