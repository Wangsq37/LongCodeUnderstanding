import unittest

import pytest

from shapely.errors import GeometryTypeError
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import shared_paths


class SharedPaths(unittest.TestCase):
    def test_shared_paths_forward(self):
        # New input: Lines with negative, zero, and float coordinates, and an overlap at edge-case positions
        g1 = LineString([(-1000, -1000.5), (0, 0), (1000, 0), (1000, 500.5)])
        g2 = LineString([(0, 0), (1000, 0), (1500, 0)])
        result = shared_paths(g1, g2)

        assert isinstance(result, GeometryCollection)
        assert len(result.geoms) == 2
        a, b = result.geoms
        assert isinstance(a, MultiLineString)
        assert len(a.geoms) == 1
        # Overlapping segment is from (0,0) to (1000,0)
        assert a.geoms[0].coords[:] == [(0, 0), (1000, 0)]
        assert b.is_empty

    def test_shared_paths_forward2(self):
        # New input: Lines with floating points and reversed direction for overlap, plus non-overlapping segments
        g1 = LineString([(2.5, -100.8), (0, 0), (10.75, 0), (10.75, 5.5)])
        g2 = LineString([(10.75, 0), (0, 0), (-50, 0)])
        result = shared_paths(g1, g2)

        assert isinstance(result, GeometryCollection)
        assert len(result.geoms) == 2
        a, b = result.geoms
        assert isinstance(b, MultiLineString)
        assert len(b.geoms) == 1
        # Shared segment is from (0,0) to (10.75,0), and actual result is [(0.0, 0.0), (10.75, 0.0)]
        assert b.geoms[0].coords[:] == [(0.0, 0.0), (10.75, 0.0)]
        assert a.is_empty

    def test_wrong_type(self):
        g1 = Point(0, 0)
        g2 = LineString([(5, 0), (15, 0)])

        with pytest.raises(GeometryTypeError):
            shared_paths(g1, g2)

        with pytest.raises(GeometryTypeError):
            shared_paths(g2, g1)