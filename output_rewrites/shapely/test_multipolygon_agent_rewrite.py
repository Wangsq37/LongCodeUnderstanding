import numpy as np
import pytest

from shapely import MultiPolygon, Polygon
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase


class TestMultiPolygon(MultiGeometryTestCase):
    def test_multipolygon(self):
        # Edge case: Empty remains, robust against empty input
        geom = MultiPolygon([])
        assert geom.is_empty
        assert len(geom.geoms) == 0

        # Complex: Polygon with very large and negative coordinates, including a negative hole
        coords = [
            (
                ((-1000000, -1000000), (0.0, 5000000), (1000000, -1000000), (5000000, 5000000), (0.0, 0.0)),
                [((-500000, -500000), (-250000, -250000), (-500000, 500000), (0.0, 0.0))],
            )
        ]
        geom = MultiPolygon(coords)
        assert isinstance(geom, MultiPolygon)
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (-1000000.0, -1000000.0),
                (0.0, 5000000.0),
                (1000000.0, -1000000.0),
                (5000000.0, 5000000.0),
                (0.0, 0.0),
                (-1000000.0, -1000000.0),
                [(-500000.0, -500000.0), (-250000.0, -250000.0), (-500000.0, 500000.0), (0.0, 0.0), (-500000.0, -500000.0)],
            ]
        ]

        # With floats and extra points, no holes
        coords2 = [(((0.0, 0.0), (0.0, 1.5), (2.7, 1.0), (1.0, 0.0), (0.0, -2.3)),)]
        geom = MultiPolygon(coords2)
        assert isinstance(geom, MultiPolygon)
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (0.0, 0.0),
                (0.0, 1.5),
                (2.7, 1.0),
                (1.0, 0.0),
                (0.0, -2.3),
                (0.0, 0.0),
            ]
        ]

        # From polygons with a degenerate hole (zero area)
        p = Polygon(
            ((10, 0), (20, 0), (20, 1), (10, 1)),
            [((15.0, 0.5), (15.0, 0.5), (15.0, 0.5))]
        )
        geom = MultiPolygon([p])
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 1.0),
                (10.0, 1.0),
                (10.0, 0.0),
                [(15.0, 0.5), (15.0, 0.5), (15.0, 0.5), (15.0, 0.5)]
            ]
        ]

        # None and empty polygons are dropped among a valid and an empty polygon
        geom_from_list_with_empty = MultiPolygon([p, None, Polygon()])
        assert geom_from_list_with_empty == geom

        # Multiple polygons with distinct coordinates
        p2 = Polygon(
            ((100, 100), (200, 100), (200, 200), (100, 200)),
            [((150, 150), (150, 175), (175, 175), (175, 150))]
        )
        geom_multiple_from_list = MultiPolygon([p, p2])
        assert len(geom_multiple_from_list.geoms) == 2
        assert [dump_coords(g) for g in geom_multiple_from_list.geoms] == [
            [
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 1.0),
                (10.0, 1.0),
                (10.0, 0.0),
                [(15.0, 0.5), (15.0, 0.5), (15.0, 0.5), (15.0, 0.5)],
            ],
            [
                (100.0, 100.0),
                (200.0, 100.0),
                (200.0, 200.0),
                (100.0, 200.0),
                (100.0, 100.0),
                [(150.0, 150.0), (150.0, 175.0), (175.0, 175.0), (175.0, 150.0), (150.0, 150.0)]
            ]
        ]

        # Or from a np.array of polygons
        geom_multiple_from_array = MultiPolygon(np.array([p, p2]))
        assert [dump_coords(g) for g in geom_multiple_from_array.geoms] == [
            [
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 1.0),
                (10.0, 1.0),
                (10.0, 0.0),
                [(15.0, 0.5), (15.0, 0.5), (15.0, 0.5), (15.0, 0.5)],
            ],
            [
                (100.0, 100.0),
                (200.0, 100.0),
                (200.0, 200.0),
                (100.0, 200.0),
                (100.0, 100.0),
                [(150.0, 150.0), (150.0, 175.0), (175.0, 175.0), (175.0, 150.0), (150.0, 150.0)]
            ]
        ]

        # Or from another multi-polygon
        geom2 = MultiPolygon(geom)
        assert len(geom2.geoms) == 1
        assert dump_coords(geom2) == [
            [
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 1.0),
                (10.0, 1.0),
                (10.0, 0.0),
                [(15.0, 0.5), (15.0, 0.5), (15.0, 0.5), (15.0, 0.5)]
            ]
        ]

        # Sub-geometry Access for polygon with zero-area hole
        assert isinstance(geom.geoms[0], Polygon)
        assert dump_coords(geom.geoms[0]) == [
            (10.0, 0.0),
            (20.0, 0.0),
            (20.0, 1.0),
            (10.0, 1.0),
            (10.0, 0.0),
            [(15.0, 0.5), (15.0, 0.5), (15.0, 0.5), (15.0, 0.5)],
        ]
        with pytest.raises(IndexError):  # index out of range
            geom.geoms[1]

        # Geo interface, non-trivial coordinates
        assert geom.__geo_interface__ == {
            "type": "MultiPolygon",
            "coordinates": [
                (
                    ((10.0, 0.0), (20.0, 0.0), (20.0, 1.0), (10.0, 1.0), (10.0, 0.0)),
                    ((15.0, 0.5), (15.0, 0.5), (15.0, 0.5), (15.0, 0.5)),
                )
            ],
        }

    def test_subgeom_access(self):
        poly0 = Polygon([(1e-10, 1e-10), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)])
        poly1 = Polygon([(0.25, 0.25), (-1e5, -1e5), (0.5, 0.5), (0.5, 0.25)])
        self.subgeom_access_test(MultiPolygon, [poly0, poly1])


def test_fail_list_of_multipolygons():
    """A list of multipolygons is not a valid multipolygon ctor argument"""
    poly = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    multi = MultiPolygon(
        [
            (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
            )
        ]
    )
    with pytest.raises(ValueError):
        MultiPolygon([multi])

    with pytest.raises(ValueError):
        MultiPolygon([poly, multi])


def test_numpy_object_array():
    geom = MultiPolygon(
        [
            (
                ((-100.5, -100.5), (0.0, 1000.1), (100.5, -100.5), (1000.1, 1000.1), (0.0, 0.0)),
                [((0.25, 1e4), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
            )
        ]
    )
    ar = np.empty(2, object)
    ar[:] = [geom, geom]
    assert ar[0] == geom
    assert ar[1] == geom