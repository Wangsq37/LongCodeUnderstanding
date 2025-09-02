import numpy as np
import pytest

from shapely import GeometryCollection, LineString, Point, wkt
from shapely.geometry import shape


@pytest.fixture()
def geometrycollection_geojson():
    # Augment with a 3D point with large coordinates, a LineString with negative coords, and a Polygon (additional type)
    return {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "Point", "coordinates": (1e10, -1e10, 9999)},
            {"type": "LineString", "coordinates": ((-1000, 0), (-500.5, -200.2), (0, 0))},
            {"type": "Polygon", "coordinates": (((0,0), (0,1), (1,1), (1,0), (0,0)),)}
        ],
    }


@pytest.mark.parametrize(
    "geom",
    [
        # Empty in various forms and edge: a geometry created from an empty np array
        GeometryCollection(),
        GeometryCollection([]),
        shape({"type": "GeometryCollection", "geometries": []}),
        wkt.loads("GEOMETRYCOLLECTION EMPTY"),
        GeometryCollection(np.array([])), 
    ],
)
def test_empty(geom):
    assert geom.geom_type == "GeometryCollection"
    assert geom.is_empty
    assert len(geom.geoms) == 0
    assert list(geom.geoms) == []


def test_empty_subgeoms():
    # Test with empty subgeometries: MultiPoint empty, empty Polygon
    from shapely.geometry import MultiPoint, Polygon
    geom = GeometryCollection([Point(), LineString(), MultiPoint(), Polygon()])
    assert geom.geom_type == "GeometryCollection"
    assert geom.is_empty
    assert len(geom.geoms) == 4
    assert list(geom.geoms) == [Point(), LineString(), MultiPoint(), Polygon()]


def test_child_with_deleted_parent():
    # test that we can remove a collection while keeping
    # children around
    a = LineString([(0, 0), (1, 1), (1, 2), (2, 2)])
    b = LineString([(0, 0), (1, 1), (2, 1), (2, 2)])
    collection = a.intersection(b)

    child = collection.geoms[0]
    # delete parent of child
    del collection

    # access geometry, this should not seg fault as 1.2.15 did
    assert child.wkt is not None


def test_from_numpy_array():
    # Use more/complex geometries, including negative and float coordinates, and empty Point
    geoms = np.array([
        Point(-999.1, 12345.6), 
        LineString([(10, 10), (20, 20), (30, 40)]),
        Point(),  # Empty
        LineString([(0.0,0.0), (0.0,1.1)]),
    ])
    geom = GeometryCollection(geoms)
    assert len(geom.geoms) == 4
    np.testing.assert_array_equal(geoms, geom.geoms)


def test_from_geojson(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    assert geom.geom_type == "GeometryCollection"
    assert len(geom.geoms) == 3

    geom_types = [g.geom_type for g in geom.geoms]
    assert "Point" in geom_types
    assert "LineString" in geom_types
    assert "Polygon" in geom_types


def test_geointerface(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    assert geom.__geo_interface__ == geometrycollection_geojson


def test_len_raises(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    with pytest.raises(TypeError):
        len(geom)


def test_numpy_object_array():
    # Use a more complex geometry in the collection, and verify storage
    complex_geom = GeometryCollection([
        LineString([(0, 0), (1e10, -1e10)]), 
        Point(42, -42),
        LineString([(3.3, 3.3), (4.4, 4.4)]),
    ])
    ar = np.empty(1, object)
    ar[:] = [complex_geom]
    assert ar[0] == complex_geom