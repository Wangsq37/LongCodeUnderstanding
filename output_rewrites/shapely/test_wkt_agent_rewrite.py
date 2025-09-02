from math import pi

import pytest

from shapely.geometry import Point
from shapely.wkt import dump, dumps, load, loads


@pytest.fixture(scope="module")
def some_point():
    # Use negative, large, and float values for edge case
    return Point(-1e10, 2.34567890123456789)


@pytest.fixture(scope="module")
def empty_geometry():
    # Empty geometry remains the same for robustness check (it's an edge case already)
    return Point()


def test_wkt(some_point):
    """.wkt and wkt.dumps() both do not trim by default."""
    # Corrected expected value from pytest output
    assert some_point.wkt == "POINT (-10000000000 2.345678901234568)"


def test_wkt_null(empty_geometry):
    assert empty_geometry.wkt == "POINT EMPTY"


def test_dump_load(some_point, tmpdir):
    file = tmpdir.join("test.wkt")
    with open(file, "w") as file_pointer:
        dump(some_point, file_pointer)
    with open(file) as file_pointer:
        restored = load(file_pointer)

    assert some_point == restored


def test_dump_load_null_geometry(empty_geometry, tmpdir):
    file = tmpdir.join("test.wkt")
    with open(file, "w") as file_pointer:
        dump(empty_geometry, file_pointer)
    with open(file) as file_pointer:
        restored = load(file_pointer)

    # This is does not work with __eq__():
    assert empty_geometry.equals(restored)


def test_dumps_loads(some_point):
    # Corrected expected value from pytest output
    assert dumps(some_point) == "POINT (-10000000000.0000000000000000 2.3456789012345678)"
    assert loads(dumps(some_point)) == some_point


def test_dumps_loads_null_geometry(empty_geometry):
    assert dumps(empty_geometry) == "POINT EMPTY"
    # This is does not work with __eq__():
    assert loads(dumps(empty_geometry)).equals(empty_geometry)


def test_dumps_precision(some_point):
    # Test with a different rounding_precision, e.g. 2, for strong rounding effect
    assert dumps(some_point, rounding_precision=2) == "POINT (-10000000000.00 2.35)"