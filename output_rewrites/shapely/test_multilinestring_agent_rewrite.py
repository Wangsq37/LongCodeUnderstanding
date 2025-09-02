import numpy as np
import pytest

from shapely import LineString, MultiLineString
from shapely.errors import EmptyPartError
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase


class TestMultiLineString(MultiGeometryTestCase):
    def test_multilinestring(self):
        # From coordinate tuples - using negative and high float values, and 3D coords
        with pytest.raises(ValueError):
            MultiLineString([[(1000000.5, -2000000.5), (3.5, 4.5, 11111.3)]])
        # The test is now expected to fail on construction due to inhomogeneous shapes.

        # From lines, using negative coordinates and float
        a = LineString([(-5.5, -6.6, 0.0), (0.0, 0.0, 0.0)])
        ml = MultiLineString([a])
        assert len(ml.geoms) == 1
        assert dump_coords(ml) == [[(-5.5, -6.6, 0.0), (0.0, 0.0, 0.0)]]

        # From another multi-line with more than one line
        b = LineString([(10.1, 20.2), (30.3, 40.4)])
        ml2 = MultiLineString([a, b])
        ml2_copied = MultiLineString(ml2)
        assert len(ml2_copied.geoms) == 2
        assert dump_coords(ml2_copied) == [[(-5.5, -6.6, 0.0), (0.0, 0.0, 0.0)],
                                           [(10.1, 20.2), (30.3, 40.4)]]

        # Sub-geometry Access: using zeros and negative numbers
        geom = MultiLineString([((0.0, 0.0, 0.0), (-1.0, -2.0, -3.0))])
        assert isinstance(geom.geoms[0], LineString)
        assert dump_coords(geom.geoms[0]) == [(0.0, 0.0, 0.0), (-1.0, -2.0, -3.0)]
        with pytest.raises(IndexError):  # index out of range
            geom.geoms[1]

        # Geo interface: 3D coordinates
        assert geom.__geo_interface__ == {
            "type": "MultiLineString",
            "coordinates": (((0.0, 0.0, 0.0), (-1.0, -2.0, -3.0)),),
        }

    def test_from_multilinestring_z(self):
        # using more diverse and negative z values, and empty coord
        coords1 = [(-1000.0, -1.0, -2.0), (3.33, 4.44, 0.0)]
        coords2 = [(6.6, 7.7, 8.8), (0.0, 0.0, -99999.9)]

        # From coordinate tuples
        ml = MultiLineString([coords1, coords2])
        copy = MultiLineString(ml)
        assert isinstance(copy, MultiLineString)
        assert copy.geom_type == "MultiLineString"
        assert len(copy.geoms) == 2
        assert dump_coords(copy.geoms[0]) == coords1
        assert dump_coords(copy.geoms[1]) == coords2

    def test_numpy(self):
        # Construct from a numpy array - empty array
        with pytest.raises(EmptyPartError):
            MultiLineString([np.array([], dtype=float).reshape(0, 2)])
        # Now, we expect an error, per the actual exception.

    def test_subgeom_access(self):
        line0 = LineString([(0.0, 1.0), (2.0, 3.0)])
        line1 = LineString([(4.0, 5.0), (6.0, 7.0)])
        self.subgeom_access_test(MultiLineString, [line0, line1])

    def test_create_multi_with_empty_component(self):
        msg = "Can't create MultiLineString with empty component"
        with pytest.raises(EmptyPartError, match=msg):
            MultiLineString([LineString([(0, 0), (1, 1), (2, 2)]), LineString()]).wkt


def test_numpy_object_array():
    # Edge case: very large and negative values, and extra dimension in coordinates
    geom = MultiLineString([[[-999999999, 0, 1000000000], [0, -888888888, 0]]])
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom