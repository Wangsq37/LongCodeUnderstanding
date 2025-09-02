import unittest

import pytest

import shapely
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import LineString, Point, Polygon


class PolylabelTestCase(unittest.TestCase):
    def test_polylabel(self):
        """
        Finds pole of inaccessibility for a polygon with a tolerance of 10

        """
        polygon = LineString(
            [(0, 0), (50, 200), (100, 100), (20, 50), (-100, -20), (-150, -200)]
        ).buffer(100)
        label = polylabel(polygon, tolerance=0.001)
        expected = Point(59.733, 111.330)
        assert expected.equals_exact(label, 1e-3)

    def test_concave_polygon(self):
        """
        Finds pole of inaccessibility for a concave polygon and ensures that
        the point is inside.

        """
        concave_polygon = LineString([(500, 0), (0, 0), (0, 500), (500, 500)]).buffer(
            100
        )
        label = polylabel(concave_polygon)
        assert concave_polygon.contains(label)

    def test_rectangle_special_case(self):
        """
        The centroid algorithm used is vulnerable to floating point errors
        and can give unexpected results for rectangular polygons. Test
        that this special case is handled correctly.
        https://github.com/mapbox/polylabel/issues/3
        """
        # New rectangle with negative coordinates and larger area (testing floating point and negative bounds)
        polygon = Polygon(
            [
                (-2000.0, -3000.0),
                (-2000.0, 2000.0),
                (3000.0, 2000.0),
                (3000.0, -3000.0),
            ]
        )
        label = polylabel(polygon)
        if shapely.geos_version >= (3, 14, 0):
            # For a rectangle: center should be ((-2000+3000)/2, (-3000+2000)/2) = (500, -500)
            assert label.coords[:] == [(500.0, -500.0)]
        elif shapely.geos_version >= (3, 12, 0):
            assert label.coords[:] == [(500.0, -500.0)]
        else:
            assert label.coords[:] == [(500.0, -500.0)]

    def test_polygon_with_hole(self):
        """
        Finds pole of inaccessibility for a polygon with a hole
        https://github.com/shapely/shapely/issues/817
        """
        # Augmented: A much larger polygon, off-origin, with a non-square hole
        polygon = Polygon(
            shell=[(100, 100), (300, 100), (300, 300), (100, 300), (100, 100)],
            holes=[[(150, 150), (250, 150), (250, 200), (150, 200), (150, 150)]],
        )
        label = polylabel(polygon, 0.05)
        assert label.x == pytest.approx(150.0)
        assert label.y == pytest.approx(250.0)

    @pytest.mark.skipif(
        shapely.geos_version < (3, 12, 0), reason="Fails with GEOS < 3.12"
    )
    def test_polygon_infinite_loop(self):
        # https://github.com/shapely/shapely/issues/1836
        # corner case that caused an infinite loop in the old custom implemetation
        # New: Degenerate polygon with large float values very close together (potential edge of precision problem)
        polygon = shapely.from_wkt(
            "POLYGON ((1e12 1e12, 1e12 1e12, 1e12 1e12, 1e12 1e12))"
        )
        label = polylabel(polygon)
        assert label.x == pytest.approx(1e12)
        assert label.y == pytest.approx(1e12)