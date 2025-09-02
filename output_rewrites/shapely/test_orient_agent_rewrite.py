import unittest

from numpy.testing import assert_array_equal

from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import orient


class OrientTestCase(unittest.TestCase):
    def test_point(self):
        # Edge case: point with float coords, large values
        point = Point(1e6, -1e6)
        assert orient(point, 1) == point
        assert orient(point, -1) == point

    def test_multipoint(self):
        # Edge case: multipoint with several points, including negative and float coordinates
        multipoint = MultiPoint([(0, 0), (-5.5, 2.2), (1e6, -1e-5), (0, 0)])
        assert orient(multipoint, 1) == multipoint
        assert orient(multipoint, -1) == multipoint

    def test_linestring(self):
        # Edge case: linestring with more points and mixed types
        linestring = LineString([(0, 0), (1, 1), (2.5, -1e3), (5, 6)])
        assert orient(linestring, 1) == linestring
        assert orient(linestring, -1) == linestring

    def test_multilinestring(self):
        # Edge case: extra linestrings, one with reversed coords, negative/zero/floats
        multilinestring = MultiLineString([
            [(0, 0), (1, 1), (2, 3)], 
            [(5, 0), (5, -5)], 
            [(-1e2, 1e2), (0, 0)]
        ])
        assert orient(multilinestring, 1) == multilinestring
        assert orient(multilinestring, -1) == multilinestring

    def test_linearring(self):
        # Edge case: LinearRing with more coordinates, including a closing duplicate, floats, and negatives
        linearring = LinearRing([(0, 0), (1, 2), (3, -1), (0, 0)])
        assert orient(linearring, 1) == linearring
        assert orient(linearring, -1) == linearring

    def test_empty_polygon(self):
        # Edge case: polygon with explicitly empty shell
        polygon = Polygon()
        assert orient(polygon) == polygon

    def test_polygon(self):
        # Edge case: polygon with 4 points, including negative and float coordinates (quadrilateral, not triangle)
        polygon = Polygon([(0.0, 0.0), (0.0, 2.5), (3.3, 2.5), (-1.1, 0.0)])
        polygon_reversed = Polygon(polygon.exterior.coords[::-1])
        assert (orient(polygon, 1)) == polygon_reversed
        assert (orient(polygon, -1)) == polygon

    def test_multipolygon(self):
        # Edge case: multipolygon with three distinct polygons
        polygon1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        polygon2 = Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])
        polygon3 = Polygon([(-5, -5), (-5, -1), (-2, -1), (-2, -5)])
        polygon1_reversed = Polygon(polygon1.exterior.coords[::-1])
        polygon2_reversed = Polygon(polygon2.exterior.coords[::-1])
        polygon3_reversed = Polygon(polygon3.exterior.coords[::-1])
        multipolygon = MultiPolygon([polygon1, polygon2, polygon3])
        assert not polygon1.exterior.is_ccw
        assert polygon2.exterior.is_ccw
        assert not polygon3.exterior.is_ccw
        assert orient(multipolygon, 1) == MultiPolygon([polygon1_reversed, polygon2, polygon3_reversed])
        assert orient(multipolygon, -1) == MultiPolygon([polygon1, polygon2_reversed, polygon3])

    def test_geometrycollection(self):
        # Edge case: GeometryCollection with multiple geometry types
        polygon = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        polygon_reversed = Polygon(polygon.exterior.coords[::-1])
        point = Point(10, 20)
        linestring = LineString([(-1, -1), (0, 0), (1.2, 3.4)])
        collection = GeometryCollection([polygon, point, linestring])
        assert orient(collection, 1) == GeometryCollection([polygon_reversed, point, linestring])
        assert orient(collection, -1) == GeometryCollection([polygon, point, linestring])

    def test_polygon_with_holes(self):
        ring_cw = LinearRing([(0, 0), (0, 1), (1, 1), (0, 0)])
        ring_cw2 = LinearRing([(0, 0), (0, 3), (3, 3), (0, 0)])
        ring_ccw = LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)])
        ring_ccw2 = LinearRing([(0, 0), (2, 2), (0, 2), (0, 0)])

        polygon_with_holes_mixed = Polygon(
            ring_ccw, [ring_cw, ring_ccw2, ring_cw2, ring_ccw]
        )
        polygon_with_holes_ccw = Polygon(
            ring_ccw, [ring_cw, ring_ccw2.reverse(), ring_cw2, ring_ccw.reverse()]
        )

        assert_array_equal(orient(polygon_with_holes_ccw, 1), polygon_with_holes_ccw)
        assert_array_equal(
            orient(polygon_with_holes_ccw, -1), polygon_with_holes_ccw.reverse()
        )
        assert_array_equal(orient(polygon_with_holes_mixed, 1), polygon_with_holes_ccw)
        assert_array_equal(
            orient(polygon_with_holes_mixed, -1), polygon_with_holes_ccw.reverse()
        )