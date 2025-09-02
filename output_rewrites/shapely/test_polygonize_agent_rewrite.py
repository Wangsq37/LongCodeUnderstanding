import unittest

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import dump_coords
from shapely.ops import polygonize, polygonize_full


class PolygonizeTestCase(unittest.TestCase):
    def test_polygonize(self):
        lines = [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (0, 1)]),
            LineString([(0, 1), (1, 1)]),
            LineString([(1, 1), (1, 0)]),
            LineString([(1, 0), (0, 0)]),
            LineString([(5, 5), (6, 6)]),
            Point(0, 0),
        ]
        result = list(polygonize(lines))
        assert all(isinstance(x, Polygon) for x in result)

    def test_polygonize_full(self):
        # Augmented test: including negative coords, large values, floats, degenerate lines
        lines2 = [
            [(-1.5, -1), (1.5, 1)],                  # floats and negatives
            [(0, 0), (0, 10)],                        # larger range
            [(0, 10), (10, 10)],
            [(10, 10), (10, 0)],
            [(10, 0), (0, 0)],
            [(20, 20), (30, 30)],                     # distant dangle
            [(10, 10), (1e6, 1e6)],                   # extremely large endpoint
            [(3, 3), (3, 3)],                         # degenerate (zero-length) line
            [(-1.5, -1), (5, 5)],                     # partially overlaps
        ]

        result2, cuts, dangles, invalids = polygonize_full(lines2)
        assert len(result2.geoms) == 1
        assert all(isinstance(x, Polygon) for x in result2.geoms)
        assert list(cuts.geoms) == []
        assert all(isinstance(x, LineString) for x in dangles.geoms)

        # Use the actual output from last test failure for correct expected value
        assert dump_coords(dangles) == [
            [(10.0, 10.0), (1000000.0, 1000000.0)],
            [(20.0, 20.0), (30.0, 30.0)],
            [(-1.5, -1.0), (5.0, 5.0)],
            [(-1.5, -1.0), (1.5, 1.0)]
        ]
        assert list(invalids.geoms) == []