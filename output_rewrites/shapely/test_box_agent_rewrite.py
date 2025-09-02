import unittest

from shapely import geometry


class BoxTestCase(unittest.TestCase):
    def test_ccw(self):
        # Augmented input: negative and float coordinates, non-square box
        b = geometry.box(-2.0, -3.5, 4.5, 7.25, ccw=True)
        assert b.exterior.coords[0] == (4.5, -3.5)
        assert b.exterior.coords[1] == (4.5, 7.25)

    def test_ccw_default(self):
        # Augmented input: large values (simulate edge case)
        b = geometry.box(1000, 2000, 10000, 15000)
        assert b.exterior.coords[0] == (10000.0, 2000.0)
        assert b.exterior.coords[1] == (10000.0, 15000.0)

    def test_cw(self):
        # Augmented input: zero-width box (xmin==xmax), vertical line segment
        b = geometry.box(5.0, -10.0, 5.0, 5.0, ccw=False)
        assert b.exterior.coords[0] == (5.0, -10.0)
        assert b.exterior.coords[1] == (5.0, 5.0)