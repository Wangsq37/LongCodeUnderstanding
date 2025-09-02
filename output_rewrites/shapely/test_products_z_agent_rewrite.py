import unittest

from shapely.geometry import LineString


class ProductZTestCase(unittest.TestCase):
    def test_line_intersection(self):
        # Modified: Diverse input lines, with floats and negative z's, also larger extents
        line1 = LineString([(-10, -10, -5.5), (10, 10, 25.5)])
        line2 = LineString([(-10, 10, 12.25), (10, -10, -1.25)])
        interxn = line1.intersection(line2)
        assert interxn.has_z
        assert interxn._ndim == 3
        assert -5.5 <= interxn.z <= 25.5