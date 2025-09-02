from collections import OrderedDict

import numpy as np
import pandas as pd

from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema

import pytest

# Credit: Polygons below come from Montreal city Open Data portal
# http://donnees.ville.montreal.qc.ca/dataset/unites-evaluation-fonciere
city_hall_boundaries = Polygon(
    (
        (-73.5541107525234, 45.5091983609661),
        (-73.5546126200639, 45.5086813829106),
        (-73.5540185061397, 45.5084409343852),
        (-73.5539986525799, 45.5084323044531),
        (-73.5535801792994, 45.5089539203786),
        (-73.5541107525234, 45.5091983609661),
    )
)
vauquelin_place = Polygon(
    (
        (-73.5542465586147, 45.5081555487952),
        (-73.5540185061397, 45.5084409343852),
        (-73.5546126200639, 45.5086813829106),
        (-73.5548825850032, 45.5084033554357),
        (-73.5542465586147, 45.5081555487952),
    )
)

city_hall_walls = [
    LineString(
        (
            (-73.5541107525234, 45.5091983609661),
            (-73.5546126200639, 45.5086813829106),
            (-73.5540185061397, 45.5084409343852),
        )
    ),
    LineString(
        (
            (-73.5539986525799, 45.5084323044531),
            (-73.5535801792994, 45.5089539203786),
            (-73.5541107525234, 45.5091983609661),
        )
    ),
]

city_hall_entrance = Point(-73.553785, 45.508722)
city_hall_balcony = Point(-73.554138, 45.509080)
city_hall_council_chamber = Point(-73.554246, 45.508931)

point_3D = Point(-73.553785, 45.508722, 300)
linestring_3D = LineString(
    (
        (-73.5541107525234, 45.5091983609661, 300),
        (-73.5546126200639, 45.5086813829106, 300),
        (-73.5540185061397, 45.5084409343852, 300),
    )
)
polygon_3D = Polygon(
    (
        (-73.5541107525234, 45.5091983609661, 300),
        (-73.5535801792994, 45.5089539203786, 300),
        (-73.5541107525234, 45.5091983609661, 300),
    )
)

def test_infer_schema_only_points():
    # Using zero, negative coordinates, and another point for edge case
    pts = [Point(0,0), Point(-1,-1), Point(1e6,1e6)]
    df = GeoDataFrame(geometry=pts)
    assert infer_schema(df) == {"geometry": "Point", "properties": OrderedDict()}

def test_infer_schema_points_and_multipoints():
    # empty multipoint, large multipoint, and a normal point
    df = GeoDataFrame(
        geometry=[
            MultiPoint([]),
            MultiPoint([Point(1e9, -1e9), Point(-1e9, 1e9), Point(0, 0)]),
            Point(3.3, 10.4),
        ]
    )
    assert infer_schema(df) == {
        "geometry": ["MultiPoint", "Point"],
        "properties": OrderedDict(),
    }

def test_infer_schema_only_multipoints():
    # MultiPoint with negative, zero, large, and float coords
    df = GeoDataFrame(
        geometry=[
            MultiPoint(
                [Point(-100, 0), Point(0, -100), Point(1e6, 1e-6), Point(3.14, 2.71)]
            )
        ]
    )
    assert infer_schema(df) == {"geometry": "MultiPoint", "properties": OrderedDict()}

def test_infer_schema_only_linestrings():
    # Edge cases: LineString with only 2 points, and one with very large coords
    ls = [
        LineString([(0, 0), (1000, 1000)]),
        LineString([(1e9, -1e9), (-1e9, 1e9), (0, 0)]),
    ]
    df = GeoDataFrame(geometry=ls)
    assert infer_schema(df) == {"geometry": "LineString", "properties": OrderedDict()}

def test_infer_schema_linestrings_and_multilinestrings():
    # MultiLineString mixing 2D and extreme values, and a simple LineString
    mls = MultiLineString([
        LineString([(0,0), (100,100)]),
        LineString([(-1e5,1e5), (1e5,-1e5)]),
    ])
    ls = LineString([(-999, 999), (500, 500)])
    df = GeoDataFrame(geometry=[mls, ls])
    assert infer_schema(df) == {
        "geometry": ["MultiLineString", "LineString"],
        "properties": OrderedDict(),
    }

def test_infer_schema_only_multilinestrings():
    # MultiLineString with more segments
    mls = MultiLineString([
        LineString([(0, 0), (2, 2), (-2, -2)]),
        LineString([(1e3, 0), (0, -1e3), (-1e3, 1e3)]),
        LineString([(10,100), (1e4,1e4)]),
    ])
    df = GeoDataFrame(geometry=[mls])
    assert infer_schema(df) == {
        "geometry": "MultiLineString",
        "properties": OrderedDict(),
    }

def test_infer_schema_only_polygons():
    # Polygon with holes and a polygon with negative values
    shell = [(-2, -2), (-2, 2), (2, 2), (2, -2), (-2, -2)]
    hole = [(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    poly_with_hole = Polygon(shell, [hole])
    poly_negative = Polygon([(-50, 0), (0, -50), (-50, -50), (-50,0)])
    df = GeoDataFrame(geometry=[poly_with_hole, poly_negative])
    assert infer_schema(df) == {"geometry": "Polygon", "properties": OrderedDict()}

def test_infer_schema_polygons_and_multipolygons():
    # MultiPolygon of polygons with holes and a normal polygon
    shell1 = [(0,0), (0,10), (10,10), (10,0), (0,0)]
    hole1 = [(2,2), (2,8), (8,8), (8,2), (2,2)]
    poly1 = Polygon(shell1, [hole1])
    shell2 = [(-10,-10), (-10,0), (0,0), (0,-10), (-10,-10)]
    poly2 = Polygon(shell2)
    mp = MultiPolygon((poly1, poly2))
    df = GeoDataFrame(geometry=[mp, poly1])
    assert infer_schema(df) == {
        "geometry": ["MultiPolygon", "Polygon"],
        "properties": OrderedDict(),
    }

def test_infer_schema_only_multipolygons():
    # MultiPolygon with three polygons, one with a hole
    shellA = [(0,0),(1,2),(2,0),(0,0)]
    shellB = [(3,3),(5,3),(5,5),(3,5),(3,3)]
    holeB = [(4,4),(4.5,4),(4.5,4.5),(4,4.5),(4,4)]
    polyA = Polygon(shellA)
    polyB = Polygon(shellB, [holeB])
    shellC = [(-9,-9), (-9,-7), (-7,-7), (-7,-9), (-9,-9)]
    polyC = Polygon(shellC)
    mp = MultiPolygon((polyA, polyB, polyC))
    df = GeoDataFrame(geometry=[mp])
    assert infer_schema(df) == {"geometry": "MultiPolygon", "properties": OrderedDict()}

def test_infer_schema_multiple_shape_types():
    # Add a 3D MultiPoint and a Point with extreme values
    mp = MultiPolygon((city_hall_boundaries, vauquelin_place))
    mls = MultiLineString(city_hall_walls)
    multi3D = MultiPoint([(1, 2, 3), (100,200,300)])
    pt_extreme = Point(-9e12, 9e12)
    df = GeoDataFrame(
        geometry=[
            mp,
            city_hall_boundaries,
            mls,
            city_hall_walls[0],
            multi3D,
            pt_extreme,
        ]
    )
    assert infer_schema(df) == {
        "geometry": [
            "3D MultiPoint",
            "MultiPolygon",
            "Polygon",
            "MultiLineString",
            "LineString",
            "Point",
        ],
        "properties": OrderedDict(),
    }

def test_infer_schema_mixed_3D_shape_type():
    # Mix even more 3D geoms, and a simple 2D point
    mp = MultiPolygon((city_hall_boundaries, vauquelin_place))
    multi3D = MultiPoint([(0,0,0), (1e5, -1e5, 135)])
    ls_2d = city_hall_walls[0]
    ls_3d = linestring_3D
    pt_2d = city_hall_balcony
    pt_3d = Point(-77.1, 48.1, -999)
    df = GeoDataFrame(
        geometry=[
            mp,
            city_hall_boundaries,
            MultiLineString(city_hall_walls),
            ls_2d,
            multi3D,
            pt_2d,
            pt_3d,
            ls_3d,
        ]
    )
    assert infer_schema(df) == {
        "geometry": [
            "3D MultiPoint",
            "3D Point",
            "3D LineString",
            "MultiPolygon",
            "Polygon",
            "MultiLineString",
            "LineString",
            "Point",
        ],
        "properties": OrderedDict(),
    }

def test_infer_schema_mixed_3D_Point():
    # Add a 3D point with different z
    pt1 = Point(10, -10, 5000)
    pt2 = Point(0, 0)
    df = GeoDataFrame(geometry=[pt2, pt1])
    assert infer_schema(df) == {
        "geometry": ["3D Point", "Point"],
        "properties": OrderedDict(),
    }

def test_infer_schema_only_3D_Points():
    # Multiple 3D points with different z
    pt1 = Point(-1, -2, -100)
    pt2 = Point(1, 2, 100)
    df = GeoDataFrame(geometry=[pt1, pt2])
    assert infer_schema(df) == {"geometry": "3D Point", "properties": OrderedDict()}

def test_infer_schema_mixed_3D_linestring():
    # 3D and 2D LineStrings, both with negative and large values
    ls_3d = LineString([(0, 0, 0), (1000, -1000, 0)])
    ls_2d = LineString([(-500, 500), (500, -500)])
    df = GeoDataFrame(geometry=[ls_2d, ls_3d])
    assert infer_schema(df) == {
        "geometry": ["3D LineString", "LineString"],
        "properties": OrderedDict(),
    }

def test_infer_schema_only_3D_linestrings():
    # Two 3D LineStrings, with different z and values
    ls1 = LineString([(0,0,0), (10,10,10)])
    ls2 = LineString([(1e6, -1e6, 1234.56), (-1e6, 1e6, -7890.12)])
    df = GeoDataFrame(geometry=[ls1, ls2])
    assert infer_schema(df) == {
        "geometry": "3D LineString",
        "properties": OrderedDict(),
    }

def test_infer_schema_mixed_3D_Polygon():
    # Polygon 2D and Polygon 3D with a hole
    shell_2d = [(0,0),(5,0),(5,5),(0,5),(0,0)]
    shell_3d = [(1,2,3),(4,5,6),(7,8,9),(1,2,3)]
    hole_3d = [(3,4,5),(4,5,6),(5,4,3),(3,4,5)]
    poly_2d = Polygon(shell_2d)
    poly_3d = Polygon(shell_3d, [hole_3d])
    df = GeoDataFrame(geometry=[poly_2d, poly_3d])
    assert infer_schema(df) == {
        "geometry": ["3D Polygon", "Polygon"],
        "properties": OrderedDict(),
    }

def test_infer_schema_only_3D_Polygons():
    # Two 3D polygons, one with a hole
    shell1 = [(-1, -1, 1), (2, -1, 1), (2, 2, 1), (-1, 2, 1), (-1,-1,1)]
    hole1 = [(0,0,1),(1,0,1),(1,1,1),(0,1,1),(0,0,1)]
    shell2 = [(5,5,5),(7,5,5),(7,7,5),(5,7,5),(5,5,5)]
    poly1 = Polygon(shell1, [hole1])
    poly2 = Polygon(shell2)
    df = GeoDataFrame(geometry=[poly1, poly2])
    assert infer_schema(df) == {"geometry": "3D Polygon", "properties": OrderedDict()}

def test_infer_schema_null_geometry_and_2D_point():
    # None and a Point at the origin
    df = GeoDataFrame(geometry=[None, Point(0,0)])
    # None geometry type is then omitted
    assert infer_schema(df) == {"geometry": "Point", "properties": OrderedDict()}

def test_infer_schema_null_geometry_and_3D_point():
    # None and a Point 3D at the origin
    df = GeoDataFrame(geometry=[None, Point(0,0,0)])
    # None geometry type is then omitted
    assert infer_schema(df) == {"geometry": "3D Point", "properties": OrderedDict()}

def test_infer_schema_null_geometry_all():
    # Both None values
    df = GeoDataFrame(geometry=[None, None])
    # None geometry type in then replaced by 'Unknown'
    assert infer_schema(df) == {"geometry": "Unknown", "properties": OrderedDict()}

@pytest.mark.parametrize(
    "array_data,dtype",
    [
        ([-2147483648, 2**31-1], np.int32),
        ([np.nan, -999999], pd.Int32Dtype()),
        ([0, 123456], np.int32),
    ]
)
def test_infer_schema_int32(array_data, dtype):
    int32col = pd.array(data=array_data, dtype=dtype)
    df = GeoDataFrame(geometry=[city_hall_entrance, city_hall_balcony])
    df["int32_column"] = int32col

    assert infer_schema(df) == {
        "geometry": "Point",
        "properties": OrderedDict([("int32_column", "int32")]),
    }

def test_infer_schema_int64():
    int64col = pd.array([0, np.nan, 2**60], dtype=pd.Int64Dtype())
    df = GeoDataFrame(geometry=[city_hall_entrance, city_hall_balcony, city_hall_council_chamber])
    df["int64_column"] = int64col

    assert infer_schema(df) == {
        "geometry": "Point",
        "properties": OrderedDict([("int64_column", "int")]),
    }