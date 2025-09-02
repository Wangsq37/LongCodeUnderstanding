import unittest

from shapely.geometry import LineString, Polygon
from shapely.ops import snap


class Snap(unittest.TestCase):
    def test_snap(self):
        # Modified input geometries for edge cases and robustness
        # Testing with a larger square, a line with negative, zero, and large coordinates, and floating points
        square = Polygon([(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000), (-1000, -1000)])
        line = LineString([(0.0, 0.0), (-999.5, -999.5), (999.3, -999.8), (1500.0, 0.0)])

        square_coords = square.exterior.coords[:]
        line_coords = line.coords[:]

        result = snap(line, square, 1.0)

        # test result is correct
        assert isinstance(result, LineString)
        # Initial guess for what the snapped coordinates will be
        assert result.coords[:] == [(0.0, 0.0), (-1000.0, -1000.0), (1000.0, -1000.0), (1500.0, 0.0)]

        # test inputs have not been modified
        assert square.exterior.coords[:] == square_coords
        assert line.coords[:] == line_coords