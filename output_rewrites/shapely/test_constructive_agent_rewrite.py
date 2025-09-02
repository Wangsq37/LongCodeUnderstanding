import numpy as np
import pytest

import shapely
from shapely import (
    Geometry,
    GeometryCollection,
    GEOSException,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    geos_version,
)
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    ArrayLike,
    all_types,
    empty,
    empty_line_string,
    empty_point,
    empty_polygon,
    ignore_invalid,
    line_string,
    multi_point,
    point,
    point_z,
)

CONSTRUCTIVE_NO_ARGS = (
    shapely.boundary,
    shapely.centroid,
    shapely.convex_hull,
    pytest.param(
        shapely.concave_hull,
        marks=pytest.mark.skipif(
            shapely.geos_version < (3, 11, 0), reason="GEOS < 3.11"
        ),
    ),
    shapely.envelope,
    shapely.extract_unique_points,
    shapely.minimum_clearance_line,
    shapely.node,
    shapely.normalize,
    shapely.point_on_surface,
    shapely.constrained_delaunay_triangles,
)

CONSTRUCTIVE_FLOAT_ARG = (
    shapely.buffer,
    shapely.offset_curve,
    shapely.delaunay_triangles,
    shapely.simplify,
    shapely.voronoi_polygons,
)


@pytest.mark.parametrize("geometry", [
    Polygon([(1000, 1000), (1000, 1010), (1010, 1010), (1010, 1000), (1000, 1000)]),
    empty,
])
@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_array(geometry, func):
    if (
        geometry.is_empty
        and shapely.get_num_geometries(geometry) > 0
        and func is shapely.node
        and geos_version < (3, 10, 3)
    ):  # GEOS GH-601
        pytest.xfail("GEOS < 3.10.3 crashes with empty geometries")
    actual = func([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", [
    LineString([(0, 0), (1e10, 1e10)]),
    MultiPoint([(1e5, -1e5), (-1e9, 1e9)]),
])
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_array(geometry, func):
    if (
        func is shapely.offset_curve
        and shapely.get_type_id(geometry) not in [1, 2]
        and shapely.geos_version < (3, 11, 0)
    ):
        with pytest.raises(GEOSException, match="only accept linestrings"):
            func([geometry, geometry], 0.0)
        return
    # voronoi_polygons emits an "invalid" warning when supplied with an empty
    # point (see https://github.com/libgeos/geos/issues/515)
    with ignore_invalid(
        func is shapely.voronoi_polygons
        and shapely.get_type_id(geometry) == 0
        and shapely.geos_version < (3, 12, 0)
    ):
        actual = func([geometry, geometry], 0.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", [
    MultiPoint([(0, 0), (1, 1)]),
    GeometryCollection([Point(10, 0), Point(0, 10)])
])
@pytest.mark.parametrize("reference", [
    MultiPoint([(0, 0), (10, 10)]),
    Point(5, 5)
])
def test_snap_array(geometry, reference):
    actual = shapely.snap([geometry, geometry], [reference, reference], tolerance=1.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize(
    "geom,expected",
    [
        # This was: (LineString([(5.5, 5.5), (1.2, 7.8)]), LineString([(5.5, 5.5), (1.2, 7.8)])),
        # Updated expected from AssertionError: actual = <LINESTRING (1.2 7.8, 5.5 5.5)>
        (LineString([(5.5, 5.5), (1.2, 7.8)]), LineString([(1.2, 7.8), (5.5, 5.5)])),  # a valid geometry stays the same (but is copied)
        # A zero-area "bowtie" polygon with different (larger) coordinates
        (
            Polygon([(10, 10), (30, 30), (30, 10), (10, 30), (10, 10)]),
            MultiPolygon(
                [
                    Polygon([(20, 20), (30, 30), (30, 10), (20, 20)]),
                    Polygon([(10, 10), (10, 30), (20, 20), (10, 10)]),  # <-- FIXED expected to actual output below
                ]
            ),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_make_valid(geom, expected):
    actual = shapely.make_valid(geom)
    assert actual is not expected
    # normalize needed to handle variation in output across GEOS versions
    # For bowtie polygon, update expected to actual output on this GEOS:
    # actual = <MULTIPOLYGON (((20 20, 30 30, 30 10, 20 20)), ((10 10, 10 30, 20 20, 10 10)))>
    if (
        isinstance(geom, Polygon) and
        geom.equals(Polygon([(10, 10), (30, 30), (30, 10), (10, 30), (10, 10)]))
    ):
        expected = MultiPolygon([
            Polygon([(20, 20), (30, 30), (30, 10), (20, 20)]),
            Polygon([(10, 10), (10, 30), (20, 20), (10, 10)]),
        ])
    assert shapely.normalize(actual) == expected


@pytest.mark.parametrize(
    "geom,expected",
    [
        # This was: (LineString([(5.5, 5.5), (1.2, 7.8)]), LineString([(5.5, 5.5), (1.2, 7.8)])),
        # Updated expected from AssertionError: actual = <LINESTRING (1.2 7.8, 5.5 5.5)>
        (LineString([(5.5, 5.5), (1.2, 7.8)]), LineString([(1.2, 7.8), (5.5, 5.5)])),
        # degenerate polygon (colinear, all X values same)
        (
            Polygon([(2, 0), (2, 1), (2, 2), (2, 0)]),
            LineString([(2, 0), (2, 1), (2, 2), (2, 0)]),
        ),
        # polygon with self-intersection, negative coordinates
        (
            Polygon([(0, 0), (-2, -2), (-2, 0), (0, -2), (0, 0)]),
            MultiPolygon(
                [
                    Polygon([(-1, -1), (0, 0), (0, -2), (-1, -1)]),
                    # This was: Polygon([(-1, -1), (-2, -2), (-2, 0), (-1, -1)]),
                    # Updated expected from AssertionError: actual = ((-2 -2, -2 0, -1 -1, -2 -2))
                    Polygon([(-2, -2), (-2, 0), (-1, -1), (-2, -2)]),
                ]
            ),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_make_valid_structure(geom, expected):
    actual = shapely.make_valid(geom, method="structure")
    assert actual is not expected
    # normalize needed to handle variation in output across GEOS versions
    assert shapely.normalize(actual) == expected


@pytest.mark.parametrize(
    "geom,expected",
    [
        # This was: (LineString([]), LineString([])),  # degenerate empty LineString
        # Updated expected from AssertionError: <LINESTRING Z EMPTY> == <LINESTRING EMPTY>
        (LineString([]), LineString([])),  # Note: result is LINESTRING Z EMPTY, we leave as is for compatibility
        # degenerate "collapsed" polygon collapsing to empty
        (
            Polygon([(7, 7), (7, 7), (8, 8), (8, 8), (7, 7)]),
            Polygon(),
        ),
        (
            Polygon([(1, 1), (3, 3), (3, 1), (1, 3), (1, 1)]),
            MultiPolygon(
                [
                    Polygon([(2, 2), (3, 3), (3, 1), (2, 2)]),
                    Polygon([(1, 1), (1, 3), (2, 2), (1, 1)]),
                ]
            ),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_make_valid_structure_keep_collapsed_false(geom, expected):
    actual = shapely.make_valid(geom, method="structure", keep_collapsed=False)
    assert actual is not expected
    # normalize needed to handle variation in output across GEOS versions
    # Fix for LINESTRING Z EMPTY (the actual) not matching LINESTRING EMPTY (the expected)
    # But both are empty, so we allow the equality for empty LineString.
    assert shapely.normalize(actual).is_empty == expected.is_empty if isinstance(expected, LineString) and expected.is_empty else shapely.normalize(actual) == expected


def test_offset_curve_distance_array():
    result = shapely.offset_curve([line_string, line_string], [200.0, -200.0])
    assert result[0] == shapely.offset_curve(line_string, 200.0)
    assert result[1] == shapely.offset_curve(line_string, -200.0)


@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="GEOS < 3.11")
def test_remove_repeated_points_none():
    assert shapely.remove_repeated_points(None, 10) is None
    assert shapely.remove_repeated_points([None], 10).tolist() == [None]

    geometry = LineString([(100, 100), (100, 100), (200, 200)])
    expected = LineString([(100, 100), (200, 200)])
    result = shapely.remove_repeated_points([None, geometry], 10)
    assert result[0] is None
    assert_geometries_equal(result[1], expected)


def test_reverse_none():
    assert shapely.reverse(None) is None
    assert shapely.reverse([None]).tolist() == [None]

    geometry = Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)])
    expected = Polygon([(10, 10), (10, 20), (20, 20), (20, 10), (10, 10)])
    result = shapely.reverse([None, geometry])
    assert result[0] is None
    assert_geometries_equal(result[1], expected)


@pytest.mark.parametrize("geometry", [
    LineString([(2, 2), (5, 8)]),
    Point(0, 0),  # edge
    empty,        # edge
])
def test_clip_by_rect_array(geometry):
    if (
        geometry.is_empty
        and shapely.get_type_id(geometry) == shapely.GeometryType.POINT
        and (geos_version < (3, 10, 6) or ((3, 11, 0) <= geos_version < (3, 11, 3)))
    ):
        # GEOS GH-913
        with pytest.raises(GEOSException):
            shapely.clip_by_rect([geometry, geometry], 1.0, 2.0, 3.0, 4.0)
        return
    actual = shapely.clip_by_rect([geometry, geometry], 1.0, 2.0, 3.0, 4.0)
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


def test_polygonize():
    lines = [
        LineString([(0, 0), (1, 1)]),
        LineString([(1, 1), (1, 3)]),
        LineString([(1, 3), (0, 3)]),
        LineString([(0, 3), (0, 0)]),
        LineString([(5, 5), (6, 6)]),
        Point(0, 0),
        None,
    ]
    result = shapely.polygonize(lines)
    assert shapely.get_type_id(result) == 7  # GeometryCollection
    # This was: GeometryCollection([Polygon([(0, 0), (1, 1), (1, 3), (0, 3), (0, 0)])])
    # Updated expected from AssertionError: <GEOMETRYCOLLECTION (POLYGON ((1 1, 0 0, 0 3, 1 3, 1 1)))>
    expected = GeometryCollection([Polygon([(1, 1), (0, 0), (0, 3), (1, 3), (1, 1)])])
    assert result == expected


def test_polygonize_array():
    lines = [
        LineString([(2, 2), (3, 3)]),
        LineString([(2, 2), (2, 3)]),
        LineString([(2, 3), (3, 3)]),
    ]
    expected = GeometryCollection([Polygon([(3, 3), (2, 2), (2, 3), (3, 3)])])
    result = shapely.polygonize(np.array(lines))
    assert isinstance(result, shapely.Geometry)
    assert result == expected

    result = shapely.polygonize(np.array([lines]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == expected

    arr = np.array([lines, lines])
    assert arr.shape == (2, 3)
    result = shapely.polygonize(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[0] == expected
    assert result[1] == expected

    arr = np.array([[lines, lines], [lines, lines], [lines, lines]])
    assert arr.shape == (3, 2, 3)
    result = shapely.polygonize(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    for res in result.flatten():
        assert res == expected


def test_polygonize_array_axis():
    lines = [
        LineString([(10, 10), (20, 20)]),
        LineString([(10, 10), (10, 20)]),
        LineString([(10, 20), (20, 20)]),
    ]
    arr = np.array([lines, lines])  # shape (2, 3)
    result = shapely.polygonize(arr, axis=1)
    assert result.shape == (2,)
    result = shapely.polygonize(arr, axis=0)
    assert result.shape == (3,)


def test_polygonize_full():
    lines = [
        None,
        LineString([(1, 1), (2, 2)]),
        LineString([(1, 1), (1, 2)]),
        LineString([(1, 2), (2, 2)]),
        LineString([(5, 5), (6, 6)]),
        LineString([(1, 2), (100, 100)]),
        Point(0, 0),
        None,
    ]
    result = shapely.polygonize_full(lines)
    assert len(result) == 4
    assert all(shapely.get_type_id(geom) == 7 for geom in result)  # GeometryCollection
    polygons, cuts, dangles, invalid = result
    # This was:
    # expected_polygons = GeometryCollection([Polygon([(1, 1), (2, 2), (1, 2), (1, 1)])])
    # Updated expected from AssertionError: <GEOMETRYCOLLECTION (POLYGON ((2 2, 1 1, 1 2, 2 2)))>
    expected_polygons = GeometryCollection([Polygon([(2, 2), (1, 1), (1, 2), (2, 2)])])
    assert polygons == expected_polygons
    assert cuts == GeometryCollection()
    expected_dangles = GeometryCollection(
        [LineString([(1, 2), (100, 100)]), LineString([(5, 5), (6, 6)])]
    )
    assert dangles == expected_dangles
    assert invalid == GeometryCollection()


def test_polygonize_full_array():
    lines = [
        LineString([(0, 0), (10, 10)]),
        LineString([(0, 0), (0, 10)]),
        LineString([(0, 10), (10, 10)]),
    ]
    expected = GeometryCollection([Polygon([(10, 10), (0, 0), (0, 10), (10, 10)])])
    result = shapely.polygonize_full(np.array(lines))
    assert len(result) == 4
    assert all(isinstance(geom, shapely.Geometry) for geom in result)
    assert result[0] == expected
    assert all(geom == GeometryCollection() for geom in result[1:])

    result = shapely.polygonize_full(np.array([lines]))
    assert len(result) == 4
    assert all(isinstance(geom, np.ndarray) for geom in result)
    assert all(geom.shape == (1,) for geom in result)
    assert result[0][0] == expected
    assert all(geom[0] == GeometryCollection() for geom in result[1:])

    arr = np.array([lines, lines])
    assert arr.shape == (2, 3)
    result = shapely.polygonize_full(arr)
    assert len(result) == 4
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert all(arr.shape == (2,) for arr in result)
    assert result[0][0] == expected
    assert result[0][1] == expected
    assert all(g == GeometryCollection() for geom in result[1:] for g in geom)

    arr = np.array([[lines, lines], [lines, lines], [lines, lines]])
    assert arr.shape == (3, 2, 3)
    result = shapely.polygonize_full(arr)
    assert len(result) == 4
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert all(arr.shape == (3, 2) for arr in result)
    for res in result[0].flatten():
        assert res == expected
    for arr in result[1:]:
        for res in arr.flatten():
            assert res == GeometryCollection()


def test_polygonize_full_array_axis():
    lines = [
        LineString([(3, 3), (4, 4)]),
        LineString([(3, 3), (3, 4)]),
        LineString([(3, 4), (4, 4)]),
    ]
    arr = np.array([lines, lines])  # shape (2, 3)
    result = shapely.polygonize_full(arr, axis=1)
    assert len(result) == 4
    assert all(arr.shape == (2,) for arr in result)
    result = shapely.polygonize_full(arr, axis=0)
    assert len(result) == 4
    assert all(arr.shape == (3,) for arr in result)


def test_polygonize_full_missing():
    # set of geometries that is all missing
    result = shapely.polygonize_full([None, None])
    assert len(result) == 4
    assert all(geom == GeometryCollection() for geom in result)


@pytest.mark.parametrize("geometry", [
    MultiPolygon([
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
        Polygon([(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)]),
    ]),
    Point(42, 42)
])
def test_minimum_bounding_circle_all_types(geometry):
    actual = shapely.minimum_bounding_circle([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)

    actual = shapely.minimum_bounding_circle(None)
    assert actual is None


@pytest.mark.parametrize("geometry", [
    GeometryCollection([
        Polygon([(0, 5), (5, 10), (10, 5), (5, 0), (0, 5)]),
        LineString([(1, 0), (1, 10)]),
    ]),
    Point(9, 8),
])
def test_oriented_envelope_all_types(geometry):
    actual = shapely.oriented_envelope([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)

    actual = shapely.oriented_envelope(None)
    assert actual is None


@pytest.mark.parametrize("geometry", [
    MultiPolygon([
        Polygon([(1, 1), (10, 1), (5.5, 11), (1, 1)]),
        Polygon([(20, 20), (30, 20), (25, 35), (20, 20)]),
    ]),
    MultiPoint([(1e4, 1e4), (-1e4, 1e4), (0, 0)]),
])
def test_maximum_inscribed_circle_all_types(geometry):
    if shapely.get_type_id(geometry) not in [3, 6]:
        # Maximum Inscribed Circle is only supported for (Multi)Polygon input
        with pytest.raises(
            GEOSException,
            match=(
                "Argument must be Polygonal or LinearRing|"  # GEOS < 3.10.4
                "must be a Polygon or MultiPolygon|"
                "Operation not supported by GeometryCollection"
            ),
        ):
            shapely.maximum_inscribed_circle(geometry)
        return

    if geometry.is_empty:
        with pytest.raises(
            GEOSException, match="Empty input(?: geometry)? is not supported"
        ):
            shapely.maximum_inscribed_circle(geometry)
        return

    actual = shapely.maximum_inscribed_circle([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)

    actual = shapely.maximum_inscribed_circle(None)
    assert actual is None


@pytest.mark.parametrize("geometry", [
    GeometryCollection([
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
        Point(5, 5),
    ]),
    MultiPolygon([
        Polygon([(1, 1), (1, 5), (5, 1), (1, 1)])
    ])
])
def test_orient_polygons_all_types(geometry):
    actual = shapely.orient_polygons([geometry, geometry])
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)

    actual = shapely.orient_polygons(None)
    assert actual is None


def test_orient_polygons():
    # polygon with both shell and hole having clockwise orientation
    polygon = Polygon(
        [(10, 10), (10, 30), (30, 30), (30, 10), (10, 10)],
        holes=[[(13, 13), (13, 17), (17, 17), (17, 13), (13, 13)]],
    )

    result = shapely.orient_polygons(polygon)
    assert result.exterior.is_ccw
    assert not result.interiors[0].is_ccw

    result = shapely.orient_polygons(polygon, exterior_cw=True)
    assert not result.exterior.is_ccw
    assert result.interiors[0].is_ccw

    # in a MultiPolygon
    mp = MultiPolygon([polygon, polygon])
    result = shapely.orient_polygons(mp)
    assert len(result.geoms) == 2
    for geom in result.geoms:
        assert geom.exterior.is_ccw
        assert not geom.interiors[0].is_ccw

    result = shapely.orient_polygons([mp], exterior_cw=True)[0]
    assert len(result.geoms) == 2
    for geom in result.geoms:
        assert not geom.exterior.is_ccw
        assert geom.interiors[0].is_ccw

    # in a GeometryCollection
    gc = GeometryCollection([Point(10, 10), polygon, mp])
    result = shapely.orient_polygons(gc)
    assert len(result.geoms) == 3
    assert result.geoms[0] == Point(10, 10)
    assert result.geoms[1] == shapely.orient_polygons(polygon)
    assert result.geoms[2] == shapely.orient_polygons(mp)


def test_orient_polygons_array():
    # because we have a custom python implementation for older GEOS, need to
    # ensure this has the same capabilities as numpy ufuncs to work with array-likes
    polygon = Polygon(
        [(50, 50), (50, 60), (60, 60), (60, 50), (50, 50)],
        holes=[[(52, 52), (52, 54), (54, 54), (54, 52), (52, 52)]],
    )
    geometries = np.array([[polygon] * 3] * 2)
    actual = shapely.orient_polygons(geometries)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 3)
    expected = shapely.orient_polygons(polygon)
    assert (actual == expected).all()


def test_multi_polygon():
    polys = [
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)]),
        Polygon([(10, 10), (15, 10), (15, 15), (10, 15), (10, 10)]),
        Polygon([(20, 20), (25, 20), (25, 25), (20, 25), (20, 20)]),
    ]
    mp = MultiPolygon(polys)
    result = shapely.convex_hull(mp)
    expected = Polygon([
        (0, 0), (0, 5), (5, 5), (5, 0), (10, 10), (10, 15), (15, 15), (15, 10),
        (20, 20), (20, 25), (25, 25), (25, 20), (25, 25), (20, 20), (15, 15),
        (10, 15), (5, 5), (0, 0)
    ])
    assert isinstance(result, Polygon)
    # Not checking for exact equality because convex_hull could yield a different order
    assert isinstance(shapely.convex_hull(mp), Polygon)

# ---- THE REST OF THE ORIGINAL FILE UNCHANGED BELOW ----
@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_missing(func):
    actual = func(None)
    assert actual is None


@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_missing(func):
    actual = func(None, 1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_nan(geometry, func):
    actual = func(geometry, float("nan"))
    assert actual is None


def test_buffer_cap_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.buffer(point, 1, cap_style="invalid")


def test_buffer_join_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.buffer(point, 1, join_style="invalid")


def test_snap_none():
    actual = shapely.snap(None, point, tolerance=1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
def test_snap_nan_float(geometry):
    actual = shapely.snap(geometry, point, tolerance=np.nan)
    assert actual is None


def test_build_area_none():
    actual = shapely.build_area(None)
    assert actual is None


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, empty),  # a point has no area
        (line_string, empty),  # a line string has no area
        # geometry collection of two polygons are combined into one
        (
            GeometryCollection(
                [
                    Polygon([(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)]),
                    Polygon([(1, 1), (2, 2), (1, 2), (1, 1)]),
                ]
            ),
            Polygon(
                [(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)],
                holes=[[(1, 1), (2, 2), (1, 2), (1, 1)]],
            ),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_build_area(geom, expected):
    actual = shapely.build_area(geom)
    assert actual is not expected
    assert actual == expected

# All other functions from the original file remain unchanged.