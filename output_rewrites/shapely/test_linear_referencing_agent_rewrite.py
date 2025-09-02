import unittest

import pytest

import shapely
from shapely.geometry import LineString, MultiLineString, Point


class LinearReferencingTestCase(unittest.TestCase):
    def setUp(self):
        self.point = Point(1, 1)
        self.line1 = LineString([(0, 0), (2, 0)])
        self.line2 = LineString([(3, 0), (3, 6)])
        self.multiline = MultiLineString(
            [list(self.line1.coords), list(self.line2.coords)]
        )

    def test_line1_project(self):
        # More comprehensive point off the line, negative and far
        point = Point(-3, 4)
        # line1 is from (0,0) to (2,0), horizontal.
        # (projected onto the line would be at x=0, y=0)
        assert self.line1.project(point) == 0.0
        assert self.line1.project(point, normalized=True) == 0.0

    def test_alias_project(self):
        # Use a point beyond the end of the line
        point = Point(4, 0)
        # line1 is from (0,0) to (2,0), so projected distance is 2 (end of line)
        assert self.line1.line_locate_point(point) == 2.0
        assert self.line1.line_locate_point(point, normalized=True) == 1.0

    def test_line2_project(self):
        # Use a point outside the extent of line2, below the starting y
        point = Point(3, -1)
        # line2 is vertical, so the projection should be 0 (start of line)
        assert self.line2.project(point) == 0.0
        assert self.line2.project(point, normalized=True) == 0.0

    def test_multiline_project(self):
        # Use a point closer to the second line, high up
        point = Point(3, 5)
        # multiline: line1 from (0,0)-(2,0), line2 from (3,0)-(3,6)
        # Should project onto line2 at (3,5), which is 2 (line1) + 5 (along line2) = 7
        # Total length = length(line1) + length(line2) = 2 + 6 = 8
        # normalized = 7/8
        assert self.multiline.project(point) == 7.0
        assert self.multiline.project(point, normalized=True) == 0.875

    def test_not_supported_project(self):
        with pytest.raises(shapely.GEOSException, match="IllegalArgumentException"):
            self.point.buffer(1.0).project(self.point)

    def test_not_on_line_project(self):
        # Points that aren't on the line project to 0.
        assert self.line1.project(Point(-10, -10)) == 0.0

    def test_line1_interpolate(self):
        assert self.line1.interpolate(0.5).equals(Point(0.5, 0.0))
        assert self.line1.interpolate(-0.5).equals(Point(1.5, 0.0))
        assert self.line1.interpolate(0.5, normalized=True).equals(Point(1, 0))
        assert self.line1.interpolate(-0.5, normalized=True).equals(Point(1, 0))

    def test_alias_interpolate(self):
        assert self.line1.line_interpolate_point(0.5).equals(Point(0.5, 0.0))
        assert self.line1.line_interpolate_point(-0.5).equals(Point(1.5, 0.0))
        assert self.line1.line_interpolate_point(0.5, normalized=True).equals(
            Point(1, 0)
        )
        assert self.line1.line_interpolate_point(-0.5, normalized=True).equals(
            Point(1, 0)
        )

    def test_line2_interpolate(self):
        assert self.line2.interpolate(0.5).equals(Point(3.0, 0.5))
        assert self.line2.interpolate(0.5, normalized=True).equals(Point(3, 3))

    def test_multiline_interpolate(self):
        assert self.multiline.interpolate(0.5).equals(Point(0.5, 0))
        assert self.multiline.interpolate(0.5, normalized=True).equals(Point(3.0, 2.0))

    def test_line_ends_interpolate(self):
        # Distances greater than length of the line or less than
        # zero yield the line's ends.
        assert self.line1.interpolate(-1000).equals(Point(0.0, 0.0))
        assert self.line1.interpolate(1000).equals(Point(2.0, 0.0))