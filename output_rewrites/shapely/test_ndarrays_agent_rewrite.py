# Tests of support for Numpy ndarrays. See
# https://github.com/sgillies/shapely/issues/26 for discussion.

import unittest
from functools import reduce

import numpy as np

from shapely import geometry


class TransposeTestCase(unittest.TestCase):
    def test_multipoint(self):
        arr = np.array([[1.0, 1.0, 2.0, 2.0, 1.0], [3.0, 4.0, 4.0, 3.0, 3.0]])
        tarr = arr.T
        shape = geometry.MultiPoint(tarr)
        coords = reduce(lambda x, y: x + y, [list(g.coords) for g in shape.geoms])
        assert coords == [(1.0, 3.0), (1.0, 4.0), (2.0, 4.0), (2.0, 3.0), (1.0, 3.0)]

    def test_linestring(self):
        # Changed input with larger, negative, and zero values
        a = np.array([
            [-10.0, 0.0, 10.0, 50.0, 100.0],
            [0.0, -10.0, 20.0, 40.0, 0.0]
        ])
        t = a.T
        s = geometry.LineString(t)
        assert list(s.coords) == [
            (-10.0, 0.0),
            (0.0, -10.0),
            (10.0, 20.0),
            (50.0, 40.0),
            (100.0, 0.0)
        ]

    def test_polygon(self):
        # Changed input with floats and a potential self-intersection
        a = np.array([
            [0.5, 1.5, -2.5, 4.5, 0.5],
            [3.8, -1.2, 3.8, 2.1, 3.8]
        ])
        t = a.T
        s = geometry.Polygon(t)
        assert list(s.exterior.coords) == [
            (0.5, 3.8),
            (1.5, -1.2),
            (-2.5, 3.8),
            (4.5, 2.1),
            (0.5, 3.8)
        ]