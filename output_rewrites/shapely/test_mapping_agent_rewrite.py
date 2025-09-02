import unittest

from shapely.geometry import Point, Polygon, mapping


class MappingTestCase(unittest.TestCase):
    def test_point(self):
        # Augmented test: Use a negative value and a very large float
        m = mapping(Point(-123456789, 1.7e+308))
        assert m["type"] == "Point"
        assert m["coordinates"] == (-123456789.0, 1.7e+308)

    def test_empty_polygon(self):
        """Empty polygons will round trip without error"""
        assert mapping(Polygon()) is not None