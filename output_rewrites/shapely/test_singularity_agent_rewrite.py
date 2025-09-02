import unittest

from shapely.geometry import Polygon


class PolygonTestCase(unittest.TestCase):
    def test_polygon_3(self):
        # NEW - diverse input: all points at (0, 0), edge case at origin
        p = (0.0, 0.0)
        poly = Polygon([p, p, p])
        assert poly.bounds == (0.0, 0.0, 0.0, 0.0)

    def test_polygon_5(self):
        # NEW - diverse input: all points at a large negative and large positive coordinate
        p = (1e10, -1e10)
        poly = Polygon([p, p, p, p, p])
        assert poly.bounds == (1e10, -1e10, 1e10, -1e10)