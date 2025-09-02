import unittest

from shapely import wkt
from shapely.geometry import shape
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import LinearRing, Polygon


class GeoThing:
    def __init__(self, d):
        self.__geo_interface__ = d


class GeoInterfaceTestCase(unittest.TestCase):
    def test_geointerface(self):
        # Convert a dictionary with negative and large coordinates
        d = {"type": "Point", "coordinates": (-9999999.0, 9999999.0)}
        geom = shape(d)
        assert geom.geom_type == "Point"
        assert tuple(geom.coords) == ((-9999999.0, 9999999.0),)

        # Convert an object that implements the geo protocol with three-dimensional coordinates
        geom = None
        thing = GeoThing({"type": "Point", "coordinates": (1.5, -2.5, 42.0)})
        geom = shape(thing)
        assert geom.geom_type == "Point"
        assert tuple(geom.coords) == ((1.5, -2.5, 42.0),)

        # Check line string with three points and mixture of floats and ints
        geom = shape({"type": "LineString", "coordinates": ((-5, 0.1), (1.0, 3), (0, -10))})
        assert isinstance(geom, LineString)
        assert tuple(geom.coords) == ((-5, 0.1), (1.0, 3), (0, -10))

        # Check linearring with repeated points and a negative ring
        geom = shape(
            {
                "type": "LinearRing",
                "coordinates": (
                    (0.0, 0.0),
                    (-1.0, 2.0),
                    (2.0, -1.0),
                    (-3.5, 0.0),
                    (0.0, 0.0),
                ),
            }
        )
        assert isinstance(geom, LinearRing)
        assert tuple(geom.coords) == (
            (0.0, 0.0),
            (-1.0, 2.0),
            (2.0, -1.0),
            (-3.5, 0.0),
            (0.0, 0.0),
        )

        # Polygon with two holes, coordinates as floats/large numbers
        geom = shape(
            {
                "type": "Polygon",
                "coordinates": (
                    ((10000.0, 10000.0), (10000.0, 10001.0), (10001.0, 10001.0), (10002.0, 9999.0), (10000.0, 10000.0)),
                    ((10000.1, 10000.1), (10000.1, 10000.2), (10000.2, 10000.2), (10000.2, 10000.1), (10000.1, 10000.1)),
                    ((9999.1, 9999.1), (9999.1, 9999.2), (9999.2, 9999.2), (9999.2, 9999.1), (9999.1, 9999.1)),
                ),
            }
        )
        assert isinstance(geom, Polygon)
        assert tuple(geom.exterior.coords) == (
            (10000.0, 10000.0),
            (10000.0, 10001.0),
            (10001.0, 10001.0),
            (10002.0, 9999.0),
            (10000.0, 10000.0),
        )
        assert len(geom.interiors) == 2

        # MultiPoint with integer, negative, float coordinates
        geom = shape({"type": "MultiPoint", "coordinates": ((0, 0), (-1.2, 3.5), (6, -7))})
        assert isinstance(geom, MultiPoint)
        assert len(geom.geoms) == 3

        # MultiLineString with two lines, one with three points
        geom = shape(
            {"type": "MultiLineString", "coordinates": (((0.0, 0.0), (1.0, 2.0)), ((-5, 10), (0, -5), (2, 2)))}
        )
        assert isinstance(geom, MultiLineString)
        assert len(geom.geoms) == 2

        # MultiPolygon with two polygons, first has a hole, second is simple
        geom = shape(
            {
                "type": "MultiPolygon",
                "coordinates": [
                    (
                        ((5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (6.0, 5.0), (5.0, 5.0)),
                        ((5.1, 5.1), (5.1, 5.2), (5.2, 5.2), (5.2, 5.1), (5.1, 5.1)),
                    ),
                    (
                        ((10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0), (10.0, 10.0)),
                        [],
                    ),
                ],
            }
        )
        assert isinstance(geom, MultiPolygon)
        assert len(geom.geoms) == 2


def test_empty_wkt_polygon():
    """Confirm fix for issue #450 - now test with explicit whitespace and lower case"""
    g = wkt.loads(" polygon empty   ")
    assert g.__geo_interface__["type"] == "Polygon"
    assert g.__geo_interface__["coordinates"] == ()


def test_empty_polygon():
    """Confirm fix for issue #450 - check polygon with 0-area single point shell"""
    g = Polygon([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
    assert g.__geo_interface__["type"] == "Polygon"
    assert g.__geo_interface__["coordinates"] == (((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),)