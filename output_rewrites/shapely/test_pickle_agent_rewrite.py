import pathlib
import pickle
import warnings
from pickle import HIGHEST_PROTOCOL, dumps, loads

import pytest

import shapely
from shapely import wkt
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

HERE = pathlib.Path(__file__).parent


# Augment TEST_DATA with additional diverse and edge case geometries
AUGMENTED_TEST_DATA = {
    "point2d_zero": Point([(0.0, 0.0)]),  # origin
    "point3d_negative": Point([(-1.0, -2.0, -3.0)]),  # negative coordinates, Z
    "linestring_large": LineString([(1e6, 1e6), (2e6, 2e6), (3e6, 3e6)]),  # large numbers
    "linearring_float": LinearRing([(0.1, 0.1), (1.1, 0.1), (1.1, 1.1), (0.1, 0.1)]),  # non-integer floats
    "polygon_empty": Polygon(),  # truly empty polygon
    "multipoint_single": MultiPoint([(10.0, 10.0)]),  # degenerate multipoint
    "multilinestring_overlap": MultiLineString([[(0,0),(10,10)], [(10,10),(20,20)]]),  # overlapping lines
    "multipolygon_negative": MultiPolygon([box(-3,-3,-2,-2), box(-1,-1,2,2)]),  # negative and positive extent
    "geometrycollection_varied": GeometryCollection([
        Point(100.0, -100.0),
        LineString([(0,0),(1,1)]), 
        Polygon([(1,2), (2,3), (3,2), (1,2)])  # triangle
    ]),
    "emptypoint": wkt.loads("POINT EMPTY"),
    "emptypolygon": wkt.loads("POLYGON EMPTY"),
    "point3d_zeroz": Point([(5.0, 7.0, 0.0)]),  # 3d with zero Z
    "polygon_with_hole": Polygon(
        [(0,0),(10,0),(10,10),(0,10),(0,0)],
        [[(3,3),(7,3),(7,7),(3,7),(3,3)]]
    ),  # polygon with a hole
}

AUGMENTED_TEST_NAMES, AUGMENTED_TEST_GEOMS = zip(*AUGMENTED_TEST_DATA.items())


@pytest.mark.parametrize("geom1", AUGMENTED_TEST_GEOMS, ids=AUGMENTED_TEST_NAMES)
def test_pickle_round_trip(geom1):
    data = dumps(geom1, HIGHEST_PROTOCOL)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geom2 = loads(data)
    assert geom2.has_z == geom1.has_z
    assert type(geom2) is type(geom1)
    assert geom2.geom_type == geom1.geom_type
    assert geom2.wkt == geom1.wkt


@pytest.mark.parametrize(
    "fname", (HERE / "data").glob("*.pickle"), ids=lambda fname: fname.name
)
def test_unpickle_pre_20(fname):
    from shapely.testing import assert_geometries_equal

    # We'll use the original TEST_DATA dict as before; the pickles on disk will match.
    geom_type = fname.name.split("_")[0]
    # Use original fallback for the on-disk pickle compatibility test
    TEST_DATA = {
        "point2d": Point([(1.0, 2.0)]),
        "point3d": Point([(1.0, 2.0, 3.0)]),
        "linestring": LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]),
        "linearring": LinearRing([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
        "polygon": Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
        "multipoint": MultiPoint([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
        "multilinestring": MultiLineString(
            [[(0.0, 0.0), (1.0, 1.0)], [(1.0, 2.0), (3.0, 3.0)]]
        ),
        "multipolygon": MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]),
        "geometrycollection": GeometryCollection([Point(1.0, 2.0), box(0, 0, 1, 1)]),
        "emptypoint": wkt.loads("POINT EMPTY"),
        "emptypolygon": wkt.loads("POLYGON EMPTY"),
    }
    expected = TEST_DATA[geom_type]

    with open(fname, "rb") as f:
        with pytest.warns(UserWarning, match="may be removed in a future version"):
            result = pickle.load(f)

    assert_geometries_equal(result, expected)


if __name__ == "__main__":
    datadir = HERE / "data"
    datadir.mkdir(exist_ok=True)

    shapely_version = shapely.__version__
    print(shapely_version)
    print(shapely.geos_version)

    # Save only original test data (so as not to break pre-2.0 unpickling logic)
    TEST_DATA = {
        "point2d": Point([(1.0, 2.0)]),
        "point3d": Point([(1.0, 2.0, 3.0)]),
        "linestring": LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]),
        "linearring": LinearRing([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
        "polygon": Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
        "multipoint": MultiPoint([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
        "multilinestring": MultiLineString(
            [[(0.0, 0.0), (1.0, 1.0)], [(1.0, 2.0), (3.0, 3.0)]]
        ),
        "multipolygon": MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]),
        "geometrycollection": GeometryCollection([Point(1.0, 2.0), box(0, 0, 1, 1)]),
        "emptypoint": wkt.loads("POINT EMPTY"),
        "emptypolygon": wkt.loads("POLYGON EMPTY"),
    }

    for name, geom in TEST_DATA.items():
        with open(datadir / f"{name}_{shapely_version}.pickle", "wb") as f:
            pickle.dump(geom, f)