import numpy as np
import pytest

from shapely import Point, geos_version
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError, UnsupportedGEOSVersionError


def test_from_coordinates():
    # Point with large integer values and negative Z
    p = Point(1000000, -500000)
    assert p.coords[:] == [(1000000.0, -500000.0)]
    assert p.has_z is False

    # PointZ with zeros
    p = Point(0.0, 0.0, -123456.789)
    assert p.coords[:] == [(0.0, 0.0, -123456.789)]
    assert p.has_z

    # empty with None coordinates (should be empty as usual)
    p = Point()
    assert p.is_empty
    assert isinstance(p.coords, CoordinateSequence)
    assert p.coords[:] == []


def test_from_sequence():
    # From single coordinate pair with negative float and big float
    p = Point((-9999.999, 12345.6789))
    assert p.coords[:] == [(-9999.999, 12345.6789)]
    p = Point([0, -1e10])
    assert p.coords[:] == [(0.0, -1e10)]

    # From coordinate sequence (XY and XYZ)
    p = Point([(7e6, 4.0)])
    assert p.coords[:] == [(7000000.0, 4.0)]
    p = Point([[0.0, -314.159]])
    assert p.coords[:] == [(0.0, -314.159)]

    # PointZ with negative and zero values
    p = Point((0.0, 3.14, -999))
    assert p.coords[:] == [(0.0, 3.14, -999.0)]
    p = Point([-1e9, 0, 987654321])
    assert p.coords[:] == [(-1e9, 0.0, 987654321.0)]
    p = Point([(2.718, -22.0, 0)])
    assert p.coords[:] == [(2.718, -22.0, 0.0)]


def test_from_numpy():
    # Construct from a numpy array with zeros
    p = Point(np.array([0.0, 0.0]))
    assert p.coords[:] == [(0.0, 0.0)]

    p = Point(np.array([-1e5, 1e-5, 2]))
    assert p.coords[:] == [(-100000.0, 0.00001, 2.0)]


def test_from_numpy_xy():
    # Construct from separate x, y numpy arrays - negative and positive
    p = Point(np.array([-2.5]), np.array([105.1]))
    assert p.coords[:] == [(-2.5, 105.1)]

    p = Point(np.array([0.0]), np.array([0.0]), np.array([-300.5]))
    assert p.coords[:] == [(0.0, 0.0, -300.5)]


def test_from_point():
    # From another point with negative XY
    p = Point(-11.0, -22.0)
    q = Point(p)
    assert q.coords[:] == [(-11.0, -22.0)]

    p = Point(0.0, -3.14, 2.718)
    q = Point(p)
    assert q.coords[:] == [(0.0, -3.14, 2.718)]


def test_from_generator():
    gen = (coord for coord in [(123456.0, -98765.0)])
    p = Point(gen)
    assert p.coords[:] == [(123456.0, -98765.0)]


def test_from_invalid():
    with pytest.raises(TypeError, match="takes at most 3 arguments"):
        Point(1, 2, 3, 4)

    # this worked in shapely 1.x, just ignoring the other coords
    with pytest.raises(
        ValueError, match="takes only scalar or 1-size vector arguments"
    ):
        Point([(2, 3), (11, 4)])


class TestPoint:
    def test_point(self):
        # Test XY point with negative float
        p = Point(-123.456, 789.012)
        assert p.x == -123.456
        assert type(p.x) is float
        assert p.y == 789.012
        assert type(p.y) is float
        assert p.coords[:] == [(-123.456, 789.012)]
        assert str(p) == p.wkt
        assert p.has_z is False
        with pytest.raises(DimensionError):
            p.z
        if geos_version >= (3, 12, 0):
            assert p.has_m is False
            with pytest.raises(DimensionError):
                p.m
        else:
            with pytest.raises(UnsupportedGEOSVersionError):
                p.m

        # Check XYZ point with all zeros
        p = Point(0.0, 0.0, 0.0)
        assert p.coords[:] == [(0.0, 0.0, 0.0)]
        assert str(p) == p.wkt
        assert p.has_z is True
        assert p.z == 0.0
        assert type(p.z) is float
        if geos_version >= (3, 12, 0):
            assert p.has_m is False
            with pytest.raises(DimensionError):
                p.m

            # TODO: Check XYM and XYZM points

        # Coordinate access, with negative values
        p = Point((-4.0, -3.0))
        assert p.x == -4.0
        assert p.y == -3.0
        assert tuple(p.coords) == ((-4.0, -3.0),)
        assert p.coords[0] == (-4.0, -3.0)
        with pytest.raises(IndexError):  # index out of range
            p.coords[1]

        # Bounds
        assert p.bounds == (-4.0, -3.0, -4.0, -3.0)

        # Geo interface
        assert p.__geo_interface__ == {"type": "Point", "coordinates": (-4.0, -3.0)}

    def test_point_empty(self):
        # Test Non-operability of Null geometry
        p_null = Point()
        assert p_null.wkt == "POINT EMPTY"
        assert p_null.coords[:] == []
        assert p_null.area == 0.0

        assert p_null.__geo_interface__ == {"type": "Point", "coordinates": ()}

    def test_coords(self):
        # From Array.txt with negative Z
        p = Point(0.0, 0.0, -1.0)
        coords = p.coords[0]
        assert coords == (0.0, 0.0, -1.0)

        # Convert to Numpy array, passing through Python sequence
        a = np.asarray(coords)
        assert a.ndim == 1
        assert a.size == 3
        assert a.shape == (3,)


def test_point_immutable():
    p = Point(3.0, 4.0)

    with pytest.raises(AttributeError):
        p.coords = (2.0, 1.0)

    with pytest.raises(TypeError):
        p.coords[0] = (2.0, 1.0)


def test_point_array_coercion():
    # don't convert to array of coordinates, keep objects with negative XY
    p = Point(-3.0, -4.0)
    arr = np.array(p)
    assert arr.ndim == 0
    assert arr.size == 1
    assert arr.dtype == np.dtype("object")
    assert arr.item() == p

    # Add another test for empty point
    p_empty = Point()
    arr_empty = np.array(p_empty)
    assert arr_empty.ndim == 0
    assert arr_empty.size == 1
    assert arr_empty.dtype == np.dtype("object")
    assert arr_empty.item() == p_empty


def test_numpy_empty_point_coords():
    pe = Point()
    # Access the coords and check for correct shape with empty input
    a = np.asarray(pe.coords)
    assert a.shape == (0, 2)


def test_numpy_object_array():
    geom = Point(0.0, 0.0)
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom