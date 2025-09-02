import unittest

from shapely.geometry import Point
from shapely.validation import explain_validity


class ValidationTestCase(unittest.TestCase):
    def test_valid(self):
        # Test with a Point with large floating-point coordinates
        # Also edge: Use negative coordinates
        assert explain_validity(Point(-1e9, 1e9)) == "Valid Geometry"