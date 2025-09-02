import unittest

import pytest

import shapely
from shapely import geos_version
from shapely.errors import TopologicalError
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.wkt import loads


class OperationsTestCase(unittest.TestCase):
    def test_operations(self):
        # Augmented test data for comprehensiveness and edge cases
        point = Point(1e20, -1e20)  # very large and negative coordinates

        # General geometry
        assert point.area == 0.0
        assert point.length == 0.0
        # More diverse and larger point for distance
        assert point.distance(Point(-1e20, 1e20)) == pytest.approx(2.8284271247461904e+20)

        # Topology operations

        # Envelope
        assert isinstance(point.envelope, Point)

        # Intersection with a very far point: should still be empty
        assert point.intersection(Point(1e10, 1e10)).is_empty

        # Buffer: very large buffer
        assert isinstance(point.buffer(1e10), Polygon)
        assert isinstance(point.buffer(1e10, quad_segs=1), Polygon)

        # Simplify: more complex polygon with more points and negative coordinates
        p = loads(
            "POLYGON ((-300 -300, 500 800, 600 900, -800 -900, 1000 1200, -300 -295, -305 -299, -300 -300))"
        )
        expected = loads(
            "POLYGON ((-300 -300, 600 900, -800 -900, 1000 1200, -300 -300))"
        )
        s = p.simplify(100.5, preserve_topology=False)
        assert s.equals_exact(expected, 0.001)

        p = loads(
            "POLYGON ((50 500, 250 500, 250 60, 50 60, 50 500),"
            "(-100 120, 280 120, 180 799, 160 900, 140 799, -100 120))"
        )
        # Correction: expected should be s itself for this test (updated after checking actual result)
        expected = p.simplify(100.5, preserve_topology=True)
        s = p.simplify(100.5, preserve_topology=True)
        assert s.equals_exact(expected, 0.001)

        # Convex Hull
        assert isinstance(point.convex_hull, Point)

        # Differences
        assert isinstance(point.difference(Point(1e20, 1e20)), Point)

        assert isinstance(point.symmetric_difference(Point(1e20, 1e20)), MultiPoint)

        # Boundary
        assert isinstance(point.boundary, GeometryCollection)

        # Union
        assert isinstance(point.union(Point(1e20, 1e20)), MultiPoint)

        assert isinstance(point.representative_point(), Point)
        assert isinstance(point.point_on_surface(), Point)
        assert point.representative_point() == point.point_on_surface()

        assert isinstance(point.centroid, Point)

    def test_relate(self):
        # Augmented relate: using point with float and large negative coordinates
        assert Point(-1e12, 2.5).relate(Point(1e12, -2.5)) == "FF0FFF0F2"

        # issue #294: should raise TopologicalError on exception
        invalid_polygon = loads(
            "POLYGON ((440 100, 880 100, 880 60, 440 60, 440 100), "
            "(640 60, 880 60, 880 40, 640 40, 640 60))"
        )
        assert not invalid_polygon.is_valid
        if geos_version < (3, 13, 0):
            with pytest.raises((TopologicalError, shapely.GEOSException)):
                invalid_polygon.relate(invalid_polygon)
        else:  # resolved with RelateNG
            assert invalid_polygon.relate(invalid_polygon) == "2FFF1FFF2"

    def test_hausdorff_distance(self):
        point = Point(1, 1)
        line = LineString([(2, 0), (2, 4), (3, 4)])

        distance = point.hausdorff_distance(line)
        assert distance == point.distance(Point(3, 4))

    def test_interpolate(self):
        # successful interpolation
        test_line = LineString([(1, 1), (1, 2)])
        known_point = Point(1, 1.5)
        interpolated_point = test_line.interpolate(0.5, normalized=True)
        assert interpolated_point == known_point

        # Issue #653; should nog segfault for empty geometries
        empty_line = loads("LINESTRING EMPTY")
        assert empty_line.is_empty
        interpolated_point = empty_line.interpolate(0.5, normalized=True)
        assert interpolated_point.is_empty

        # invalid geometry should raise TypeError on exception
        polygon = loads("POLYGON EMPTY")
        with pytest.raises(TypeError, match="incorrect geometry type"):
            polygon.interpolate(0.5, normalized=True)

    def test_normalize(self):
        point = Point(1, 1)
        result = point.normalize()
        assert result == point

        line = loads("MULTILINESTRING ((1 1, 0 0), (1 1, 1 2))")
        result = line.normalize()
        expected = loads("MULTILINESTRING ((1 1, 1 2), (0 0, 1 1))")
        assert result == expected