import binascii
import math
import struct
import sys

import pytest

from shapely import geos_version, wkt
from shapely.geometry import Point
from shapely.tests.legacy.conftest import shapely20_todo
from shapely.wkb import dump, dumps, load, loads


@pytest.fixture(scope="module")
def some_point():
    # Use points with larger, negative, and zero coordinates, and z-dim.
    return Point(-12345.6789, 0.0, 987654321)


def bin2hex(value):
    return binascii.b2a_hex(value).upper().decode("utf-8")


def hex2bin(value):
    return binascii.a2b_hex(value)


def hostorder(fmt, value):
    """Re-pack a hex WKB value to native endianness if needed

    This routine does not understand WKB format, so it must be provided a
    struct module format string, without initial indicator character ("@=<>!"),
    which will be interpreted as big- or little-endian with standard sizes
    depending on the endian flag in the first byte of the value.
    """

    if fmt and fmt[0] in "@=<>!":
        raise ValueError("Initial indicator character, one of @=<>!, in fmt")
    if not fmt or fmt[0] not in "cbB":
        raise ValueError("Missing endian flag in fmt")

    (hexendian,) = struct.unpack(fmt[0], hex2bin(value[:2]))
    hexorder = {0: ">", 1: "<"}[hexendian]
    sysorder = {"little": "<", "big": ">"}[sys.byteorder]
    if hexorder == sysorder:
        return value  # Nothing to do

    return bin2hex(
        struct.pack(
            sysorder + fmt,
            {">": 0, "<": 1}[sysorder],
            *struct.unpack(hexorder + fmt, hex2bin(value))[1:],
        )
    )


@pytest.mark.skipif(geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_dumps_srid(some_point):
    # Now uses a Z point, with negative and large coordinates, to check SRID handling.
    result = dumps(some_point)
    expected = "0101000080A1F831E6D61CC8C0000000000000000000008058346FCD41"
    assert bin2hex(result) == expected
    result = dumps(some_point, srid=3857)
    expected = "01010000A0110F0000A1F831E6D61CC8C0000000000000000000008058346FCD41"
    assert bin2hex(result) == expected


def test_dumps_endianness(some_point):
    # Now uses a Z point and explicit endianness, including negative and large.
    result = dumps(some_point)
    expected = "0101000080A1F831E6D61CC8C0000000000000000000008058346FCD41"
    assert bin2hex(result) == expected
    result = dumps(some_point, big_endian=False)
    expected = "0101000080A1F831E6D61CC8C0000000000000000000008058346FCD41"
    assert bin2hex(result) == expected
    result = dumps(some_point, big_endian=True)
    expected = "0080000001C0C81CD6E631F8A1000000000000000041CD6F3458800000"
    assert bin2hex(result) == expected


def test_dumps_hex(some_point):
    # Change to hex dump for 3D point with negative and large values.
    result = dumps(some_point, hex=True)
    expected = "0101000080A1F831E6D61CC8C0000000000000000000008058346FCD41"
    assert result == expected


@pytest.mark.skipif(geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_loads_srid():
    # load a geometry which includes an srid and Z value
    geom = loads(hex2bin(
        "01010000A0E6100000B2B4339A1E859CC00000000000000000294C7E8ACD9B6341"
    ))
    assert isinstance(geom, Point)
    assert geom.coords[:] == [(-1825.279885108856, 0.0, 10280556.327917175)]
    # by default srid is not exported
    result = dumps(geom)
    expected = hostorder(
        "BIddd", "0101000080B2B4339A1E859CC00000000000000000294C7E8ACD9B6341"
    )
    assert bin2hex(result) == expected
    # include the srid in the output
    result = dumps(geom, include_srid=True)
    expected = hostorder(
        "BIIddd",
        "01010000A0E6100000B2B4339A1E859CC00000000000000000294C7E8ACD9B6341"
    )
    assert bin2hex(result) == expected
    # replace geometry srid with another (using a big SRID)
    result = dumps(geom, srid=999999999)
    expected = hostorder(
        "BIIddd",
        "01010000A0FFC99A3BB2B4339A1E859CC00000000000000000294C7E8ACD9B6341"
    )
    assert bin2hex(result) == expected


def test_loads_hex(some_point):
    # Use a Z point as input here too, round-trip from hex
    wkb_hex = dumps(some_point, hex=True)
    # Test loads via hex representation for 3D
    assert loads(wkb_hex, hex=True) == some_point


def test_dump_load_binary(some_point, tmpdir):
    file = tmpdir.join("test.wkb")
    with open(file, "wb") as file_pointer:
        dump(some_point, file_pointer)
    with open(file, "rb") as file_pointer:
        restored = load(file_pointer)

    assert some_point == restored


def test_dump_load_hex(some_point, tmpdir):
    file = tmpdir.join("test.wkb")
    with open(file, "w") as file_pointer:
        dump(some_point, file_pointer, hex=True)
    with open(file) as file_pointer:
        restored = load(file_pointer, hex=True)

    assert some_point == restored


# pygeos handles both bytes and str
@shapely20_todo
def test_dump_hex_load_binary(some_point, tmpdir):
    """Asserts that reading a binary file as text (hex mode) fails."""
    file = tmpdir.join("test.wkb")
    with open(file, "w") as file_pointer:
        dump(some_point, file_pointer, hex=True)

    with pytest.raises(TypeError):
        with open(file, "rb") as file_pointer:
            load(file_pointer)


def test_dump_binary_load_hex(some_point, tmpdir):
    """Asserts that reading a text file (hex mode) as binary fails."""
    file = tmpdir.join("test.wkb")
    with open(file, "wb") as file_pointer:
        dump(some_point, file_pointer)

    # TODO(shapely-2.0) on windows this doesn't seem to error with pygeos,
    # but you get back a point with garbage coordinates
    if sys.platform == "win32":
        with open(file) as file_pointer:
            restored = load(file_pointer, hex=True)
        assert some_point != restored
        return

    with pytest.raises((UnicodeEncodeError, UnicodeDecodeError)):
        with open(file) as file_pointer:
            load(file_pointer, hex=True)


def test_point_empty():
    # Use a 2D POINT EMPTY (normal) but follow the edge, also try -999.0 values
    g = wkt.loads("POINT EMPTY")
    result = dumps(g, big_endian=False)
    # Use math.isnan for second part of the WKB representation (there are
    # many byte representations for NaN)
    assert result[: -2 * 8] == b"\x01\x01\x00\x00\x00"
    coords = struct.unpack("<2d", result[-2 * 8 :])
    assert len(coords) == 2
    assert all(math.isnan(val) for val in coords)


@pytest.mark.skipif(geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_point_z_empty():
    # Check for Z EMPTY with the altered hostorder
    g = wkt.loads("POINT Z EMPTY")
    assert g.wkb_hex == hostorder(
        "BIddd", "0101000080000000000000F87F000000000000F87F000000000000F87F"
    )