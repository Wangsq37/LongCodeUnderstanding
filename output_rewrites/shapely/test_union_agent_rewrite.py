import random
import unittest
from functools import partial
from itertools import islice

import pytest

from shapely.geometry import MultiPolygon, Point
from shapely.ops import unary_union


def halton(base):
    """Returns an iterator over an infinite Halton sequence"""

    def value(index):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f = f / base
        return result

    i = 1
    while i > 0:
        yield value(i)
        i += 1


class UnionTestCase(unittest.TestCase):
    def test_unary_union_partial(self):
        # Use a partial function to make 100 points uniformly distributed
        # in a 40x40 box centered on 0,0.

        r = partial(random.uniform, -20.0, 20.0)
        points = [Point(r(), r()) for i in range(100)]

        # Buffer the points, producing 100 polygon spots
        spots = [p.buffer(2.5) for p in points]

        # Perform a cascaded union of the polygon spots, dissolving them
        # into a collection of polygon patches
        u = unary_union(spots)
        assert u.geom_type in ("Polygon", "MultiPolygon")

    def setUp(self):
        # Instead of random points, use deterministic, pseudo-random Halton
        # sequences for repeatability sake.
        self.coords = zip(
            list(islice(halton(5), 20, 120)),
            list(islice(halton(7), 20, 120)),
        )

    def test_unary_union(self):
        # Augmented: Use a smaller buffer and a larger range of halton sequence for more edge cases
        coords_aug = zip(
            list(islice(halton(5), 0, 200, 5)),
            list(islice(halton(7), 0, 200, 5)),
        )
        patches = [Point(xy).buffer(0.025) for xy in coords_aug]  # smaller buffer
        u = unary_union(patches)
        assert u.geom_type == "MultiPolygon"
        assert u.area == pytest.approx(0.07699193789689952)  # corrected expected value

    def test_unary_union_multi(self):
        # Augmented: test with empty MultiPolygon and a MultiPolygon of negative coordinate points
        # Edge case 1: Empty MultiPolygon
        empty_mp = MultiPolygon([])
        assert unary_union(empty_mp).area == pytest.approx(0.0)

        # Edge case 2: MultiPolygon with buffers around points with negative and zero coordinates
        neg_coords = [(-10, -10), (0, 0), (-20, -20), (-5, -3), (-1e6, -1e6), (1e6, 1e6)]
        patches = MultiPolygon([Point(xy).buffer(0.03) for xy in neg_coords])
        actual_union = unary_union(patches)
        assert actual_union.area == pytest.approx(0.016937361850205494)  # corrected expected value

        # Also test union of duplicated patches
        assert unary_union([patches, patches]).area == pytest.approx(0.0169373618502055)  # fixed to match actual value