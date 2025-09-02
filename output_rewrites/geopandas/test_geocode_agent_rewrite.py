import pandas as pd

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries
from geopandas._compat import HAS_PYPROJ
from geopandas.tools import geocode, reverse_geocode
from geopandas.tools.geocoding import _prepare_geocode_result

import pytest
from geopandas.testing import assert_geodataframe_equal
from geopandas.tests.util import assert_geoseries_equal, mock
from pandas.testing import assert_series_equal

geopy = pytest.importorskip("geopy")


class ForwardMock(mock.MagicMock):
    """
    Mock the forward geocoding function.
    Returns the passed in address and (p, p+.5) where p increases
    at each call

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n = 0.0

    def __call__(self, *args, **kwargs):
        self.return_value = args[0], (self._n, self._n + 0.5)
        self._n += 1
        return super().__call__(*args, **kwargs)


class ReverseMock(mock.MagicMock):
    """
    Mock the reverse geocoding function.
    Returns the passed in point and 'address{p}' where p increases
    at each call

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n = 0

    def __call__(self, *args, **kwargs):
        self.return_value = f"address{self._n}", args[0]
        self._n += 1
        return super().__call__(*args, **kwargs)


@pytest.fixture
def locations():
    # More diverse test input -- very long, empty, and unicode address
    locations = [
        "",
        "1600 Amphitheatre Parkway, Mountain View, CA",
        "𝓣𝓱𝓲𝓼 𝓲𝓼 unicode test 🚗"
    ]
    return locations


@pytest.fixture
def points():
    # Test with negative, zero, very large, float coordinates, and repeated points
    points = [
        Point(0, 0),
        Point(-180, 85.05112878),  # lon min, lat max for Web Mercator 
        Point(179.9999, -85.05112878),  # lon max, lat min
        Point(123456789.123, -987654321.321),  # very large/small float
        Point(0, 0)
    ]
    return points


def test_prepare_result():
    # Augmented: Use edge and extreme point values, including empty string
    p0 = Point(0, 0)
    p1 = Point(179.9999, -85.05112878)
    p2 = Point(-180, 85.05112878)
    d = {
        "empty": ("", p0.coords[0]),
        "max_ll": ("address_max", p1.coords[0]),
        "min_ll": ("address_min", p2.coords[0])
    }

    df = _prepare_geocode_result(d)
    assert type(df) is GeoDataFrame
    if HAS_PYPROJ:
        assert df.crs == "EPSG:4326"
    assert len(df) == 3
    assert "address" in df

    coords = df.loc["empty"]["geometry"].coords[0]
    test = p0.coords[0]
    # Output from the df should be lon/lat
    assert coords[0] == pytest.approx(test[1])
    assert coords[1] == pytest.approx(test[0])
    assert df.loc["empty"]["address"] == ""

    coords = df.loc["max_ll"]["geometry"].coords[0]
    test = p1.coords[0]
    assert coords[0] == pytest.approx(test[1])
    assert coords[1] == pytest.approx(test[0])
    assert df.loc["max_ll"]["address"] == "address_max"

    coords = df.loc["min_ll"]["geometry"].coords[0]
    test = p2.coords[0]
    assert coords[0] == pytest.approx(test[1])
    assert coords[1] == pytest.approx(test[0])
    assert df.loc["min_ll"]["address"] == "address_min"


def test_prepare_result_none():
    # Use a mixture of missing values and large values
    p0 = Point(123456789.123, -987654321.321)
    d = {"nan": (None, None), "huge": ("address_huge", p0.coords[0])}

    df = _prepare_geocode_result(d)
    assert type(df) is GeoDataFrame
    if HAS_PYPROJ:
        assert df.crs == "EPSG:4326"
    assert len(df) == 2
    assert "address" in df

    row = df.loc["nan"]
    # Should be missing geometry and NaN address for None
    assert len(row["geometry"].coords) == 0
    assert row["geometry"].is_empty
    assert pd.isna(row["address"])

    # Check huge address
    row_huge = df.loc["huge"]
    assert row_huge["address"] == "address_huge"
    test = p0.coords[0]
    coords = row_huge["geometry"].coords[0]
    # Output from the df should be lon/lat order
    assert coords[0] == pytest.approx(test[1])
    assert coords[1] == pytest.approx(test[0])


@pytest.mark.parametrize("geocode_result", (None, (None, None)))
def test_prepare_geocode_result_when_result_is(geocode_result):
    # Try with float-index key and None value
    result = {3.14: geocode_result}
    # Fix: Set the expected output to have index [3.14], matching actual output
    expected_output = GeoDataFrame(
        {"geometry": [Point()], "address": [None]},
        index=[3.14],
        crs="EPSG:4326",
    )

    output = _prepare_geocode_result(result)

    assert_geodataframe_equal(output, expected_output)


def test_bad_provider_forward():
    from geopy.exc import GeocoderNotFound

    with pytest.raises(GeocoderNotFound):
        geocode(["cambridge, ma"], "badprovider")


def test_bad_provider_reverse():
    from geopy.exc import GeocoderNotFound

    with pytest.raises(GeocoderNotFound):
        reverse_geocode([Point(0, 0)], "badprovider")


def test_forward(locations, points):
    from geopy.geocoders import Photon

    # Use the locations fixture which has more diverse test cases now
    for provider in ["photon", Photon]:
        with mock.patch("geopy.geocoders.Photon.geocode", ForwardMock()) as m:
            g = geocode(locations, provider=provider, timeout=2)
            assert len(locations) == m.call_count

        n = len(locations)
        assert isinstance(g, GeoDataFrame)

        # Output should be Point(float(x) + 0.5, float(x)) for each input
        expected = GeoSeries(
            [Point(float(x) + 0.5, float(x)) for x in range(n)], crs="EPSG:4326"
        )
        assert_geoseries_equal(expected, g["geometry"])
        # Address should be the input locations
        assert_series_equal(g["address"], pd.Series(locations, name="address"))


def test_reverse(locations, points):
    from geopy.geocoders import Photon

    # Use the points fixture which has edge/duplicate/large cases
    for provider in ["photon", Photon]:
        with mock.patch("geopy.geocoders.Photon.reverse", ReverseMock()) as m:
            g = reverse_geocode(points, provider=provider, timeout=2)
            assert len(points) == m.call_count

        assert isinstance(g, GeoDataFrame)

        expected = GeoSeries(points, crs="EPSG:4326")
        assert_geoseries_equal(expected, g["geometry"])
        address = pd.Series(
            ["address" + str(x) for x in range(len(points))], name="address"
        )
        assert_series_equal(g["address"], address)