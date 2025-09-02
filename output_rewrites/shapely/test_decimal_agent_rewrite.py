from decimal import Decimal

import pytest

from shapely import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

# Augmented test data: diverse cases including negative, zero, large, fractional, and some edge cases
items2d = [
    [(0.0, -1000.0), (1e10, 120.5), (-140.5, 0.0), (0.0, -1000.0)],  # negatives, large, floats
    [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],  # all zero
    [(123456789.123, 987654321.987), (1.23, 4.56), (-7.89, 0.01), (123456789.123, 987654321.987)],  # large floats, negative, small
]

items2d_mixed = [
    [
        (Decimal("0.0"), Decimal("-1000.0")),
        (Decimal("1E+10"), 120.5),
        (-140.5, Decimal("0.0")),
        (0.0, -1000.0),
    ],
    [
        (Decimal("0.0"), Decimal("0.0")),
        (Decimal("0.0"), 0.0),
        (0.0, Decimal("0.0")),
        (0.0, 0.0),
    ],
    [
        (Decimal("123456789.123"), Decimal("987654321.987")),
        (Decimal("1.23"), 4.56),
        (-7.89, Decimal("0.01")),
        (123456789.123, 987654321.987),
    ],
]

items2d_decimal = [
    [
        (Decimal("0.0"), Decimal("-1000.0")),
        (Decimal("1E+10"), Decimal("120.5")),
        (Decimal("-140.5"), Decimal("0.0")),
        (Decimal("0.0"), Decimal("-1000.0")),
    ],
    [
        (Decimal("0.0"), Decimal("0.0")),
        (Decimal("0.0"), Decimal("0.0")),
        (Decimal("0.0"), Decimal("0.0")),
        (Decimal("0.0"), Decimal("0.0")),
    ],
    [
        (Decimal("123456789.123"), Decimal("987654321.987")),
        (Decimal("1.23"), Decimal("4.56")),
        (Decimal("-7.89"), Decimal("0.01")),
        (Decimal("123456789.123"), Decimal("987654321.987")),
    ],
]

items3d = [
    [(0.0, -1000.0, 0), (1e10, 120.5, -999999999), (-140.5, 0.0, 999999999), (0.0, -1000.0, 0)],
    [(0.0, 0.0, 0), (0.0, 0.0, 0), (0.0, 0.0, 0), (0.0, 0.0, 0)],
    [(123456789.123, 987654321.987, 123456789), (1.23, 4.56, -7), (-7.89, 0.01, 0), (123456789.123, 987654321.987, 123456789)],
]

items3d_mixed = [
    [
        (Decimal("0.0"), Decimal("-1000.0"), Decimal(0)),
        (Decimal("1E+10"), 120.5, Decimal(-999999999)),
        (-140.5, Decimal("0.0"), 999999999),
        (0.0, -1000.0, 0),
    ],
    [
        (Decimal("0.0"), Decimal("0.0"), Decimal(0)),
        (Decimal("0.0"), 0.0, 0),
        (0.0, Decimal("0.0"), 0),
        (0.0, 0.0, 0),
    ],
    [
        (Decimal("123456789.123"), Decimal("987654321.987"), Decimal(123456789)),
        (Decimal("1.23"), 4.56, -7),
        (-7.89, Decimal("0.01"), Decimal(0)),
        (123456789.123, 987654321.987, 123456789),
    ],
]

items3d_decimal = [
    [
        (Decimal("0.0"), Decimal("-1000.0"), Decimal(0)),
        (Decimal("1E+10"), Decimal("120.5"), Decimal(-999999999)),
        (Decimal("-140.5"), Decimal("0.0"), Decimal(999999999)),
        (Decimal("0.0"), Decimal("-1000.0"), Decimal(0)),
    ],
    [
        (Decimal("0.0"), Decimal("0.0"), Decimal(0)),
        (Decimal("0.0"), Decimal("0.0"), Decimal(0)),
        (Decimal("0.0"), Decimal("0.0"), Decimal(0)),
        (Decimal("0.0"), Decimal("0.0"), Decimal(0)),
    ],
    [
        (Decimal("123456789.123"), Decimal("987654321.987"), Decimal(123456789)),
        (Decimal("1.23"), Decimal("4.56"), Decimal(-7)),
        (Decimal("-7.89"), Decimal("0.01"), Decimal(0)),
        (Decimal("123456789.123"), Decimal("987654321.987"), Decimal(123456789)),
    ],
]

all_geoms = [
    [
        Point(items[0][0]),
        Point(*items[0][0]),
        MultiPoint(items[0]),
        LinearRing(items[0]),
        LineString(items[0]),
        MultiLineString(items),
        Polygon(items[0]),
        MultiPolygon(
            [
                Polygon(items[1]),
                Polygon(items[0], holes=items[1:]),
            ]
        ),
        GeometryCollection([Point(items[0][0]), Polygon(items[0])]),
    ]
    for items in [
        items2d,
        items2d_mixed,
        items2d_decimal,
        items3d,
        items3d_mixed,
        items3d_decimal,
    ]
]

@pytest.mark.parametrize("geoms", list(zip(*all_geoms)))
def test_decimal(geoms):
    assert geoms[0] == geoms[1] == geoms[2]
    assert geoms[3] == geoms[4] == geoms[5]