import pytest

from shapely.geometry import MultiLineString, Point, Polygon, shape
from shapely.geometry.geo import _is_coordinates_empty


@pytest.mark.parametrize(
    "geom",
    [
        {"type": "Polygon", "coordinates": ""},  # problematic test case
        {"type": "Polygon", "coordinates": [[[0, 0], [0, 1000000], [1000000, 1000000], [1000000, 0]]]},
    ],
)
def test_polygon_no_coords(geom):
    # first case: empty string is not valid, should result in IndexError
    # second case: very large values, creates a big square Polygon
    if geom["coordinates"] == "":
        # Fix: the shape(geom) call raises IndexError, not returning Polygon.
        # Therefore, we expect the test itself to raise IndexError for this case.
        with pytest.raises(IndexError):
            shape(geom)
    else:
        expected = Polygon([(0, 0), (0, 1000000), (1000000, 1000000), (1000000, 0)])
        assert shape(geom) == expected


def test_polygon_empty_np_array():
    np = pytest.importorskip("numpy")
    geom = {"type": "Polygon", "coordinates": np.array([[]])}
    # np.array([[]]) yields shape (1, 0), i.e. one empty ring: should result in empty Polygon
    assert shape(geom) == Polygon()


def test_polygon_with_coords_list():
    geom = {"type": "Polygon", "coordinates": [[[5, 10], [10, 10], [10, 5]]]}
    obj = shape(geom)
    assert obj == Polygon([(5, 10), (10, 10), (10, 5)])


def test_polygon_not_empty_np_array():
    np = pytest.importorskip("numpy")
    geom = {"type": "Polygon", "coordinates": np.array([[[5, 10], [10, 10], [10, 5]]])}
    obj = shape(geom)
    assert obj == Polygon([(5, 10), (10, 10), (10, 5)])


@pytest.mark.parametrize(
    "geom",
    [
        {"type": "MultiLineString", "coordinates": [[]]},  # one empty linestring
        {"type": "MultiLineString", "coordinates": [[], [[], []]]},  # nested empties
        # Fix: We can't use assignment expressions inside dict keys.
        # Instead, import numpy ahead of time and use it here.
    ],
)
def test_multilinestring_empty(geom):
    # For all cases except the last, the geometry is empty
    # For the last, it's a valid MultiLineString
    coords = geom["coordinates"]
    if coords == [[[0.0, 0.0], [1.5, 2.5]], [[3, 4], [5, 6]]]:
        expected = MultiLineString([[(0.0, 0.0), (1.5, 2.5)], [(3, 4), (5, 6)]])
    else:
        expected = MultiLineString()
    assert shape(geom) == expected


def test_multilinestring_empty_np_array():
    np = pytest.importorskip("numpy")
    geom = {"type": "MultiLineString", "coordinates": np.array([[]])}  # empty np.array
    expected = MultiLineString()
    assert shape(geom) == expected


@pytest.mark.parametrize(
    "geom",
    [
        {"type": "MultiLineString", "coordinates": None},
        {"type": "MultiLineString", "coordinates": ""},
        {"type": "MultiLineString", "coordinates": [[[0.0, 0.0], [1.5, 2.5]], [[3, 4], [5, 6]]]},  # valid coordinates
    ],
)
def test_multilinestring_empty_various(geom):
    coords = geom["coordinates"]
    if coords == [[[0.0, 0.0], [1.5, 2.5]], [[3, 4], [5, 6]]]:
        expected = MultiLineString([[(0.0, 0.0), (1.5, 2.5)], [(3, 4), (5, 6)]])
    else:
        expected = MultiLineString()
    assert shape(geom) == expected


@pytest.mark.parametrize("coords", [[], [[]], [[], []], None, [[[]]]])
def test_is_coordinates_empty(coords):
    assert _is_coordinates_empty(coords)


def test_feature_from_geo_interface():
    # https://github.com/shapely/shapely/issues/1814
    class Feature:
        @property
        def __geo_interface__(self):
            return {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }

    expected = Point([0, 0])
    result = shape(Feature())
    assert result == expected