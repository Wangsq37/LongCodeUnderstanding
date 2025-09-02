import unittest

import pytest

from shapely import geometry
from shapely.ops import transform


class IdentityTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def func(self, x, y, z=None):
        return tuple(c for c in [x, y, z] if c)

    def test_empty(self):
        g = geometry.Point()
        h = transform(self.func, g)
        assert h.is_empty

    def test_point(self):
        # Augmented: negative and float coordinates, edge value
        g = geometry.Point(-1000.5, 0.0)
        h = transform(self.func, g)
        assert h.geom_type == "Point"
        assert list(h.coords) == [(-1000.5, 0.0)]

    def test_line(self):
        # Augmented: zero and large negative, float coord
        g = geometry.LineString([(-5000.0, 0), (123456.789, -99999.1)])
        h = transform(self.func, g)
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(-5000.0, 0), (123456.789, -99999.1)]

    def test_linearring(self):
        # Augmented: closed ring, negative floats, zeros
        g = geometry.LinearRing([(0, 0), (-1.5, 2.5), (-3.5, -1.5), (0, 0)])
        h = transform(self.func, g)
        assert h.geom_type == "LinearRing"
        assert list(h.coords) == [(0, 0), (-1.5, 2.5), (-3.5, -1.5), (0, 0)]

    def test_polygon(self):
        # Augmented: buffer a point far from origin, big radius
        g = geometry.Point(10000, -10000).buffer(1000.0)
        h = transform(self.func, g)
        assert h.geom_type == "Polygon"
        assert g.area == pytest.approx(h.area)

    def test_multipolygon(self):
        # Augmented: multipoint with large negative and positive coords
        g = geometry.MultiPoint([(-1e6, 2e6), (1e6, -2e6)]).buffer(50000.0)
        h = transform(self.func, g)
        assert h.geom_type == "MultiPolygon"
        assert g.area == pytest.approx(h.area)


class LambdaTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def test_point(self):
        # Augmented: apply translation, input with large negative and float value
        g = geometry.Point(-200.5, 1000.0)
        h = transform(lambda x, y, z=None: (x + 10.0, y - 25.5), g)
        assert h.geom_type == "Point"
        assert list(h.coords) == [(-190.5, 974.5)]

    def test_line(self):
        # Augmented: big float translation, negative input coords
        g = geometry.LineString([(10.0, 20.0), (-5.0, -15.0)])
        h = transform(lambda x, y, z=None: (x - 100.5, y + 2000.0), g)
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(-90.5, 2020.0), (-105.5, 1985.0)]

    def test_polygon(self):
        # Augmented: buffer near origin, then translation
        g = geometry.Point(100.0, -100.0).buffer(25.0)
        h = transform(lambda x, y, z=None: (x * 2, y + 100), g)
        assert h.geom_type == "Polygon"
        assert g.area == pytest.approx(1960.3428065912124)
        assert h.centroid.x == pytest.approx(200.0)
        assert h.centroid.y == pytest.approx(0.0)

    def test_multipolygon(self):
        # Augmented: buffer far points, transform with multiplication and offset
        g = geometry.MultiPoint([(-1000, 1000), (2000, -2000)]).buffer(100.0)
        h = transform(lambda x, y, z=None: (x * -2, abs(y) + 100), g)
        assert h.geom_type == "MultiPolygon"
        assert g.area == pytest.approx(62730.9698109188)
        assert h.centroid.x == pytest.approx(-1000.0)
        assert h.centroid.y == pytest.approx(1600.0)