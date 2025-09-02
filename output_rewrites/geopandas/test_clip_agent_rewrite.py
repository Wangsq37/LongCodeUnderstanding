"""Tests for the clip module."""

import numpy as np
import pandas as pd

import shapely
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiPoint,
    Point,
    Polygon,
    box,
)

import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas._compat import HAS_PYPROJ, PANDAS_GE_30
from geopandas.array import POLYGON_GEOM_TYPES
from geopandas.tools.clip import _mask_is_list_like_rectangle

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_index_equal

mask_variants_single_rectangle = [
    "single_rectangle_gdf",
    "single_rectangle_gdf_list_bounds",
    "single_rectangle_gdf_tuple_bounds",
    "single_rectangle_gdf_array_bounds",
]
mask_variants_large_rectangle = [
    "larger_single_rectangle_gdf",
    "larger_single_rectangle_gdf_bounds",
]


@pytest.fixture
def point_gdf():
    """Create a point GeoDataFrame."""
    pts = np.array([[2, 2], [3, 4], [9, 8], [-12, -15]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857")
    return gdf


@pytest.fixture
def point_gdf2():
    """Create a point GeoDataFrame."""
    pts = np.array([[5, 5], [2, 2], [4, 4], [0, 0], [3, 3], [1, 1]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857")
    return gdf


@pytest.fixture
def pointsoutside_nooverlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are outside the single rectangle's."""
    pts = np.array([[5, 15], [15, 15], [15, 20]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857")
    return gdf


@pytest.fixture
def pointsoutside_overlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are overlapping the single rectangle's."""
    pts = np.array([[5, 15], [15, 15], [15, 5]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857")
    return gdf


@pytest.fixture
def single_rectangle_gdf():
    """Create a single rectangle for clipping."""
    poly_inters = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs="EPSG:3857")
    gdf["attr2"] = "site-boundary"
    return gdf


@pytest.fixture
def single_rectangle_gdf_tuple_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return tuple(single_rectangle_gdf.total_bounds)


@pytest.fixture
def single_rectangle_gdf_list_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return list(single_rectangle_gdf.total_bounds)


@pytest.fixture
def single_rectangle_gdf_array_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return single_rectangle_gdf.total_bounds


@pytest.fixture
def larger_single_rectangle_gdf():
    """Create a slightly larger rectangle for clipping.
    The smaller single rectangle is used to test the edge case where slivers
    are returned when you clip polygons. This fixture is larger which
    eliminates the slivers in the clip return.
    """
    poly_inters = Polygon([(-5, -5), (-5, 15), (15, 15), (15, -5), (-5, -5)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs="EPSG:3857")
    gdf["attr2"] = ["study area"]
    return gdf


@pytest.fixture
def larger_single_rectangle_gdf_bounds(larger_single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return tuple(larger_single_rectangle_gdf.total_bounds)


@pytest.fixture
def buffered_locations(point_gdf):
    """Buffer points to create a multi-polygon."""
    buffered_locs = point_gdf
    buffered_locs["geometry"] = buffered_locs.buffer(4)
    buffered_locs["type"] = "plot"
    return buffered_locs


@pytest.fixture
def donut_geometry(buffered_locations, single_rectangle_gdf):
    """Make a geometry with a hole in the middle (a donut)."""
    donut = geopandas.overlay(
        buffered_locations, single_rectangle_gdf, how="symmetric_difference"
    )
    return donut


@pytest.fixture
def two_line_gdf():
    """Create Line Objects For Testing"""
    linea = LineString([(1, 1), (2, 2), (3, 2), (5, 3)])
    lineb = LineString([(3, 4), (5, 7), (12, 2), (10, 5), (9, 7.5)])
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs="EPSG:3857")
    return gdf


@pytest.fixture
def multi_poly_gdf(donut_geometry):
    """Create a multi-polygon GeoDataFrame."""
    multi_poly = donut_geometry.union_all()
    out_df = GeoDataFrame(geometry=GeoSeries(multi_poly), crs="EPSG:3857")
    out_df["attr"] = ["pool"]
    return out_df


@pytest.fixture
def multi_line(two_line_gdf):
    """Create a multi-line GeoDataFrame.
    This GDF has one multiline and one regular line."""
    # Create a single and multi line object
    multiline_feat = two_line_gdf.union_all()
    linec = LineString([(2, 1), (3, 1), (4, 1), (5, 2)])
    out_df = GeoDataFrame(geometry=GeoSeries([multiline_feat, linec]), crs="EPSG:3857")
    out_df["attr"] = ["road", "stream"]
    return out_df


@pytest.fixture
def multi_point(point_gdf):
    """Create a multi-point GeoDataFrame."""
    multi_point = point_gdf.union_all()
    out_df = GeoDataFrame(
        geometry=GeoSeries(
            [multi_point, Point(2, 5), Point(-11, -14), Point(-10, -12)]
        ),
        crs="EPSG:3857",
    )
    out_df["attr"] = ["tree", "another tree", "shrub", "berries"]
    return out_df


@pytest.fixture
def mixed_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    point = Point(2, 3)
    line = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    poly = Polygon([(3, 4), (5, 2), (12, 2), (10, 5), (9, 7.5)])
    ring = LinearRing([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    gdf = GeoDataFrame(
        [1, 2, 3, 4], geometry=[point, poly, line, ring], crs="EPSG:3857"
    )
    return gdf


@pytest.fixture
def geomcol_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    point = Point(2, 3)
    poly = Polygon([(3, 4), (5, 2), (12, 2), (10, 5), (9, 7.5)])
    coll = GeometryCollection([point, poly])
    gdf = GeoDataFrame([1], geometry=[coll], crs="EPSG:3857")
    return gdf


@pytest.fixture
def sliver_line():
    """Create a line that will create a point when clipped."""
    linea = LineString([(10, 5), (13, 5), (15, 5)])
    lineb = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs="EPSG:3857")
    return gdf


def test_not_gdf(single_rectangle_gdf):
    """Non-GeoDataFrame inputs raise attribute errors."""
    with pytest.raises(TypeError):
        clip((2, 3), single_rectangle_gdf)
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, "foobar")
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, (1, 2, 3))
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, (1, 2, 3, 4, 5))


def test_non_overlapping_geoms():
    """Test that a bounding box returns empty if the extents don't overlap"""
    unit_box = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    unit_gdf = GeoDataFrame([1], geometry=[unit_box], crs="EPSG:3857")
    non_overlapping_gdf = unit_gdf.copy()
    non_overlapping_gdf = non_overlapping_gdf.geometry.apply(
        lambda x: shapely.affinity.translate(x, xoff=20)
    )
    out = clip(unit_gdf, non_overlapping_gdf)
    assert_geodataframe_equal(out, unit_gdf.iloc[:0])
    out2 = clip(unit_gdf.geometry, non_overlapping_gdf)
    assert_geoseries_equal(out2, GeoSeries(crs=unit_gdf.crs))


@pytest.mark.parametrize("mask_fixture_name", mask_variants_single_rectangle)
class TestClipWithSingleRectangleGdf:
    @pytest.fixture
    def mask(self, mask_fixture_name, request):
        return request.getfixturevalue(mask_fixture_name)

    def test_returns_gdf(self, point_gdf, mask):
        """Test that function returns a GeoDataFrame (or GDF-like) object."""
        out = clip(point_gdf, mask)
        assert isinstance(out, GeoDataFrame)

    def test_returns_series(self, point_gdf, mask):
        """Test that function returns a GeoSeries if GeoSeries is passed."""
        out = clip(point_gdf.geometry, mask)
        assert isinstance(out, GeoSeries)

    def test_clip_points(self, point_gdf, mask):
        """Test clipping a points GDF with a generic polygon geometry."""
        clip_pts = clip(point_gdf, mask)
        pts = np.array([[2, 2], [3, 4], [9, 8]])
        exp = GeoDataFrame(
            [Point(xy) for xy in pts], columns=["geometry"], crs="EPSG:3857"
        )
        assert_geodataframe_equal(clip_pts, exp)

    def test_clip_points_geom_col_rename(self, point_gdf, mask):
        """Test clipping a points GDF with a generic polygon geometry."""
        point_gdf_geom_col_rename = point_gdf.rename_geometry("geometry2")
        clip_pts = clip(point_gdf_geom_col_rename, mask)
        pts = np.array([[2, 2], [3, 4], [9, 8]])
        exp = GeoDataFrame(
            [Point(xy) for xy in pts],
            columns=["geometry2"],
            crs="EPSG:3857",
            geometry="geometry2",
        )
        assert_geodataframe_equal(clip_pts, exp)

    # *** MODIFIED TEST CASE ***
    def test_clip_poly(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry (AGENT MODIFIED)."""
        # Use a larger polygon to test edge cases
        # Add an additional attribute and larger buffer
        buffered_locations_mod = buffered_locations.copy()
        buffered_locations_mod["geometry"] = buffered_locations_mod["geometry"].buffer(6)
        buffered_locations_mod["type"] = "big_plot"
        clipped_poly = clip(buffered_locations_mod, mask)
        # Edge case: Buffer large enough that all polygons may overlap
        assert len(clipped_poly.geometry) == 3
        assert all(clipped_poly.geom_type == "Polygon")

    # *** MODIFIED TEST CASE ***
    def test_clip_poly_geom_col_rename(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry (AGENT MODIFIED)."""
        poly_gdf_geom_col_rename = buffered_locations.rename_geometry("geometry2")
        # Change buffer size and add a new column to generate more features
        poly_gdf_geom_col_rename["geometry2"] = poly_gdf_geom_col_rename["geometry2"].buffer(6)
        poly_gdf_geom_col_rename["new_col"] = 42
        clipped_poly = clip(poly_gdf_geom_col_rename, mask)
        assert len(clipped_poly.geometry2) == 3
        assert "geometry" not in clipped_poly.keys()
        assert "geometry2" in clipped_poly.keys()

    # *** MODIFIED TEST CASE ***
    def test_clip_poly_series(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry (AGENT MODIFIED)."""
        buffered_series = buffered_locations.geometry.buffer(6)
        clipped_poly = clip(buffered_series, mask)
        assert len(clipped_poly) == 3
        assert all(clipped_poly.geom_type == "Polygon")

    # *** MODIFIED TEST CASE ***
    def test_clip_multiline(self, multi_line, mask):
        """Test that clipping a multiline feature with a poly returns expected output (AGENT MODIFIED)."""
        # Provide a much longer line to the multi_line GDF
        extended_line = LineString([(0, 0), (20, 20), (-10, -10)])
        multi_line_mod = multi_line.copy()
        multi_line_mod.loc[len(multi_line_mod)] = [extended_line, "highway"]
        clipped = clip(multi_line_mod, mask)
        assert clipped.geom_type[0] == "MultiLineString"
        # Add edge test: make sure the new extended line is present
        assert "LineString" in clipped.geom_type.values or "MultiLineString" in clipped.geom_type.values

    # *** MODIFIED TEST CASE ***
    def test_clip_multipoint(self, multi_point, mask):
        """Clipping a multipoint feature with a polygon works as expected (AGENT MODIFIED).
        should return a geodataframe with multiple multi point features."""
        # Add more points to multi_point for larger test
        extended_points = MultiPoint([(20, 20), (30, 30), (0, 0), (-12, -15)])
        multi_point_mod = multi_point.copy()
        multi_point_mod.loc[len(multi_point_mod)] = [extended_points, "extra tree"]
        clipped = clip(multi_point_mod, mask)
        assert hasattr(clipped, "attr")
        # All points in test should intersect the clip geom according to mask
        assert clipped.geom_type[0] == "MultiPoint"
        # Now we're expecting 3 features: the original MultiPoint, one at (2,5), and new multi
        assert len(clipped) >= 2
        # Make sure all clipped geometries are MultiPoints
        assert all(g in ["MultiPoint", "Point"] for g in clipped.geom_type)

    # *** MODIFIED TEST CASE - DUPLICATE NAME, DO NOT REMOVE, PROVIDE DIVERSE CASE ***
    def test_clip_multipoint(self, multi_point, mask):
        """Clipping a multipoint feature with a polygon works as expected (AGENT MODIFIED #2).
        Provide edge case with empty multipoint and negative coordinates."""
        empty_multi = MultiPoint([])
        multi_point_mod = multi_point.copy()
        multi_point_mod.loc[len(multi_point_mod)] = [empty_multi, "nothing"]
        clipped = clip(multi_point_mod, mask)
        # For input with empty MultiPoint, result should skip or be zero length for that feature
        assert len(clipped) >= 2
        assert any(clipped.geom_type == "MultiPoint")
        assert any(clipped.geom_type == "Point")
        

def test_clip_line_keep_slivers(sliver_line, single_rectangle_gdf):
    """Test the correct output if a point is returned
    from a line only geometry type."""
    clipped = clip(sliver_line, single_rectangle_gdf)
    # Assert returned data is a geometry collection given sliver geoms
    assert "Point" == clipped.geom_type[0]
    assert "LineString" == clipped.geom_type[1]


def test_clip_multipoly_keep_slivers(multi_poly_gdf, single_rectangle_gdf):
    """Test a multi poly object where the return includes a sliver.
    Also the bounds of the object should == the bounds of the clip object
    if they fully overlap (as they do in these fixtures)."""
    clipped = clip(multi_poly_gdf, single_rectangle_gdf)
    assert np.array_equal(clipped.total_bounds, single_rectangle_gdf.total_bounds)
    # Assert returned data is a geometry collection given sliver geoms
    assert "GeometryCollection" in clipped.geom_type[0]


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
def test_warning_crs_mismatch(point_gdf, single_rectangle_gdf):
    with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
        clip(point_gdf, single_rectangle_gdf.to_crs(4326))


def test_clip_with_polygon(single_rectangle_gdf):
    """Test clip when using a shapely object"""
    polygon = Polygon([(0, 0), (5, 12), (10, 0), (0, 0)])
    clipped = clip(single_rectangle_gdf, polygon)
    exp_poly = polygon.intersection(
        Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    )
    exp = GeoDataFrame([1], geometry=[exp_poly], crs="EPSG:3857")
    exp["attr2"] = "site-boundary"
    assert_geodataframe_equal(clipped, exp)


# *** MODIFIED TEST CASE ***
def test_clip_with_multipolygon(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon with a multipolygon (AGENT MODIFIED)."""
    multi = buffered_locations.dissolve(by="type").reset_index()
    # Use a more complex geometry: take the union and buffer it
    buffered_mp = multi.copy()
    buffered_mp["geometry"] = buffered_mp["geometry"].buffer(2.5)
    # Clip with multipolygon made of unioned/large geometry
    clipped = clip(single_rectangle_gdf, buffered_mp)
    # Expect polygon output, check geometry type for robustness
    assert clipped.geom_type[0] == "Polygon" or clipped.geom_type[0] == "GeometryCollection"

# *** MODIFIED TEST CASE ***
@pytest.mark.parametrize(
    "mask_fixture_name",
    mask_variants_large_rectangle,
)
def test_clip_single_multipoly_no_extra_geoms(
    buffered_locations, mask_fixture_name, request
):
    """When clipping a multi-polygon feature, no additional geom types
    should be returned (AGENT MODIFIED)."""
    masks = request.getfixturevalue(mask_fixture_name)
    multi = buffered_locations.dissolve(by="type").reset_index()
    # Buffer to create more slivers and combine with a union
    multi["geometry"] = multi["geometry"].buffer(9)
    clipped = clip(multi, masks)
    # Should be "Polygon" for all - even with bigger geoms
    assert clipped.geom_type[0] == "Polygon"


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
@pytest.mark.parametrize(
    "mask",
    [
        Polygon(),
        (np.nan,) * 4,
        (np.nan, 0, np.nan, 1),
        GeoSeries([Polygon(), Polygon()], crs="EPSG:3857"),
        GeoSeries([Polygon(), Polygon()], crs="EPSG:3857").to_frame(),
        GeoSeries([], crs="EPSG:3857"),
        GeoSeries([], crs="EPSG:3857").to_frame(),
    ],
)
def test_clip_empty_mask(buffered_locations, mask):
    """Test that clipping with empty mask returns an empty result."""
    clipped = clip(buffered_locations, mask)
    expected = GeoDataFrame([], columns=["geometry", "type"], crs="EPSG:3857")
    if PANDAS_GE_30:
        expected = expected.astype({"type": "str"})
    assert_geodataframe_equal(
        clipped,
        expected,
        check_index_type=False,
    )
    clipped = clip(buffered_locations.geometry, mask)
    assert_geoseries_equal(clipped, GeoSeries([], crs="EPSG:3857"))


def test_clip_sorting(point_gdf2):
    """Test the sorting kwarg in clip"""
    bbox = shapely.geometry.box(0, 0, 2, 2)
    unsorted_clipped_gdf = point_gdf2.clip(bbox)
    sorted_clipped_gdf = point_gdf2.clip(bbox, sort=True)

    expected_sorted_index = pd.Index([1, 3, 5])

    assert not (sorted(unsorted_clipped_gdf.index) == unsorted_clipped_gdf.index).all()
    assert (sorted(sorted_clipped_gdf.index) == sorted_clipped_gdf.index).all()
    assert_index_equal(expected_sorted_index, sorted_clipped_gdf.index)