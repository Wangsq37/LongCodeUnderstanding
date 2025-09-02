import numpy as np
import pytest

from shapely import LineString, geos_version
from shapely.tests.common import (
    line_string,
    line_string_m,
    line_string_z,
    line_string_zm,
    point,
    point_m,
    point_z,
    point_zm,
)


class TestCoords:
    """
    Shapely assumes contiguous C-order float64 data for internal ops.
    Data should be converted to contiguous float64 if numpy exists.
    c9a0707 broke this a little bit.
    """

    def test_data_promotion(self):
        coords = np.array([[0, -99999999], [2.123456, 99999999]], dtype=np.float32)
        processed_coords = np.array(LineString(coords).coords)

        assert coords.tolist() == processed_coords.tolist()

    def test_data_destriding(self):
        coords = np.array([[-10000.01, 34.56], [789.99, 0]], dtype=np.float32)

        # Easy way to introduce striding: reverse list order
        processed_coords = np.array(LineString(coords[::-1]).coords)

        assert coords[::-1].tolist() == processed_coords.tolist()


class TestCoordsGetItem:
    def test_index_coords(self):
        c = [
            (1e9, -1e9),
            (-1e5, 1e5),
            (0.0, 0.0),
            (-9999.99, 9999.99)
        ]
        g = LineString(c)
        for i in range(-4, 4):
            assert g.coords[i] == c[i]
        with pytest.raises(IndexError):
            g.coords[4]
        with pytest.raises(IndexError):
            g.coords[-5]

    def test_index_coords_z(self):
        c = [
            (1e9, -1e9, 0.123),
            (-1e5, 1e5, -0.456),
            (0.0, 0.0, 9999999.9),
            (-9999.99, 9999.99, -999.99)
        ]
        g = LineString(c)
        for i in range(-4, 4):
            assert g.coords[i] == c[i]
        with pytest.raises(IndexError):
            g.coords[4]
        with pytest.raises(IndexError):
            g.coords[-5]

    def test_index_coords_misc(self):
        g = LineString()  # empty
        with pytest.raises(IndexError):
            g.coords[0]
        with pytest.raises(TypeError):
            g.coords[0.0]

    def test_slice_coords(self):
        c = [
            (3.1, 2.8),
            (-9999.99, 9999.99),
            (0.0, 0.0),
            (1e-12, -1e-12),
            (1, 2)
        ]
        g = LineString(c)
        assert g.coords[1:] == c[1:]
        assert g.coords[:-1] == c[:-1]
        assert g.coords[::-1] == c[::-1]
        assert g.coords[::2] == c[::2]
        assert g.coords[:5] == c[:5]
        assert g.coords[5:] == c[5:] == []

    def test_slice_coords_z(self):
        c = [
            (3.1, 2.8, 77.7),
            (-9999.99, 9999.99, -123.123),
            (0.0, 0.0, 0.0),
            (1e-12, -1e-12, 0.0),
            (1, 2, 3)
        ]
        g = LineString(c)
        assert g.coords[1:] == c[1:]
        assert g.coords[:-1] == c[:-1]
        assert g.coords[::-1] == c[::-1]
        assert g.coords[::2] == c[::2]
        assert g.coords[:5] == c[:5]
        assert g.coords[5:] == c[5:] == []


class TestXY:
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def test_arrays(self):
        # float and negative, large range
        x, y = LineString([(1e6, -1e6), (0.0, 0.0), (-1e-6, 1e-6)]).xy
        assert len(x) == 3
        assert list(x) == [1e6, 0.0, -1e-6]
        assert len(y) == 3
        assert list(y) == [-1e6, 0.0, 1e-6]


@pytest.mark.parametrize("geom", [point, point_z, line_string, line_string_z])
def test_coords_array_copy(geom):
    """Test CoordinateSequence.__array__ method."""
    coord_seq = geom.coords
    assert np.array(coord_seq) is not np.array(coord_seq)
    assert np.array(coord_seq, copy=True) is not np.array(coord_seq, copy=True)

    # Behaviour of copy=False is different between NumPy 1.x and 2.x
    if int(np.version.short_version.split(".", 1)[0]) >= 2:
        with pytest.raises(ValueError, match="A copy is always created"):
            np.array(coord_seq, copy=False)
    else:
        assert np.array(coord_seq, copy=False) is np.array(coord_seq, copy=False)


@pytest.mark.skipif(geos_version < (3, 12, 0), reason="GEOS < 3.12")
def test_coords_with_m():
    # Use edge-case values including zeros, negatives, large, and float
    pm = LineString([(0.0, 0.0, -1.0), (1e10, -1e10, 1.5)])
    pzm = LineString([(1234.5678, -8765.4321, 0.99, -0.01)])
    lsm = LineString([
        (-1e9, 1e9, 1.0),
        (0.0, 0.0, -100.1),
        (99999.999, -99999.999, 0.0),
    ])
    lszm = LineString([
        (1e12, -1e12, 4.2, 1.0),
        (0.0, 0.0, -4.2, 2.34),
        (-7777.77, 7777.77, 0.0, -99.9),
    ])

    assert pm.coords[:] == [(0.0, 0.0, -1.0), (1e10, -1e10, 1.5)]
    assert pzm.coords[:] == [(1234.5678, -8765.4321, 0.99, -0.01)]
    assert lsm.coords[:] == [
        (-1e9, 1e9, 1.0),
        (0.0, 0.0, -100.1),
        (99999.999, -99999.999, 0.0),
    ]
    assert lszm.coords[:] == [
        (1e12, -1e12, 4.2, 1.0),
        (0.0, 0.0, -4.2, 2.34),
        (-7777.77, 7777.77, 0.0, -99.9),
    ]