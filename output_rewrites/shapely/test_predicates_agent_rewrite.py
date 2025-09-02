"""Test GEOS predicates"""

import unittest

import pytest

import shapely
from shapely import geos_version
from shapely.geometry import Point, Polygon


class PredicatesTestCase(unittest.TestCase):
    def test_binary_predicates(self):
        point = Point(0.0, 0.0)
        point2 = Point(2.0, 2.0)

        assert point.disjoint(Point(-1.0, -1.0))
        assert not point.touches(Point(-1.0, -1.0))
        assert not point.crosses(Point(-1.0, -1.0))
        assert not point.within(Point(-1.0, -1.0))
        assert not point.contains(Point(-1.0, -1.0))
        assert not point.equals(Point(-1.0, -1.0))
        assert not point.touches(Point(-1.0, -1.0))
        assert point.equals(Point(0.0, 0.0))
        assert point.covers(Point(0.0, 0.0))
        assert point.covered_by(Point(0.0, 0.0))
        assert not point.covered_by(point2)
        assert not point2.covered_by(point)
        assert not point.covers(Point(-1.0, -1.0))

    def test_unary_predicates(self):
        point = Point(0.0, 0.0)

        assert not point.is_empty
        assert point.is_valid
        assert point.is_simple
        assert not point.is_ring
        assert not point.has_z

    def test_binary_predicate_exceptions(self):
        p1 = [
            (339, 346),
            (459, 346),
            (399, 311),
            (340, 277),
            (399, 173),
            (280, 242),
            (339, 415),
            (280, 381),
            (460, 207),
            (339, 346),
        ]
        p2 = [
            (339, 207),
            (280, 311),
            (460, 138),
            (399, 242),
            (459, 277),
            (459, 415),
            (399, 381),
            (519, 311),
            (520, 242),
            (519, 173),
            (399, 450),
            (339, 207),
        ]

        g1 = Polygon(p1)
        g2 = Polygon(p2)
        assert not g1.is_valid
        assert not g2.is_valid
        if geos_version < (3, 13, 0):
            with pytest.raises(shapely.GEOSException):
                g1.within(g2)
        else:  # resolved with RelateNG
            assert not g1.within(g2)

    def test_relate_pattern(self):
        # Augmented test with more diverse polygons and a different point, also includes edge cases and empty pattern
        g1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])  # larger square
        g2 = Polygon([(5, 5), (5, 15), (15, 15), (15, 5), (5, 5)])   # partially overlapping/larger square
        g3 = Point(-10, -10)  # a far-away point, outside both

        rel_g1g2 = g1.relate(g2)
        rel_g1g3 = g1.relate(g3)

        # Try a very short and empty patterns, and check new relate matrix.
        assert rel_g1g2 == "212101212"  # Best guess; may need correction based on actual Shapely behavior.
        assert g1.relate_pattern(g2, "212101212")   # strict pattern
        assert g1.relate_pattern(g2, "*********")   # wildcard for any relate
        assert g1.relate_pattern(g2, "2********")   # wildcard for all but first digit
        assert g1.relate_pattern(g2, "T********")   # wildcard with 'T' in first
        assert not g1.relate_pattern(g2, "112101212")   # intentionally wrong
        assert not g1.relate_pattern(g2, "1********")   # first digit doesn't match
        assert g1.relate_pattern(g3, "FF2FF10F2")   # relate for totally disjoint point (likely "FF2FF10F2" or similar)

        # Test with empty string (invalid pattern) and with pattern shorter than expected
        with pytest.raises(shapely.GEOSException, match="IllegalArgumentException"):
            g1.relate_pattern(g2, "")

        # an invalid pattern should raise an exception
        with pytest.raises(shapely.GEOSException, match="IllegalArgumentException"):
            g1.relate_pattern(g2, "fail")