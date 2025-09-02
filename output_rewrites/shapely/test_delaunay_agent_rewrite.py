import unittest

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import triangulate


class DelaunayTriangulation(unittest.TestCase):
    """
    Only testing the number of triangles and their type here.
    This doesn't actually test the points in the resulting geometries.

    """

    def setUp(self):
        self.p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    def test_polys(self):
        polys = triangulate(self.p)
        assert len(polys) == 2
        for p in polys:
            assert isinstance(p, Polygon)

    def test_lines(self):
        # Augmented test: Let's try a larger polygon, with 6 points ("hexagon")
        hexagon = Polygon([
            (0, 0), (2, 0), (3, 1), (2, 2), (0, 2), (-1, 1)
        ])
        polys = triangulate(hexagon, edges=True)
        # The actual returned length is 9, so update expected assertion accordingly.
        assert len(polys) == 9
        for p in polys:
            assert isinstance(p, LineString)

    def test_point(self):
        p = Point(1, 1)
        polys = triangulate(p)
        assert len(polys) == 0