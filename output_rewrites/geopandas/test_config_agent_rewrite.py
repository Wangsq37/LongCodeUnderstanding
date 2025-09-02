import geopandas

import pytest


def test_options():
    # More comprehensive check for repr: test for different options
    assert "io_engine: " in repr(geopandas.options)

    # Test for a superset (should still match exactly, as before)
    assert set(dir(geopandas.options)) == {
        "display_precision",
        "use_pygeos",
        "io_engine",
    }

    # Try accessing a truly odd option name to check AttributeError
    with pytest.raises(AttributeError):
        geopandas.options._private_attribute

    # Try setting a float to a non-existing option, to diversify the input
    with pytest.raises(AttributeError):
        geopandas.options.non_existing_option = 1.234


def test_options_display_precision():
    # Edge case: default should be None
    assert geopandas.options.display_precision is None

    # Use larger integer and test with float assignment as well
    geopandas.options.display_precision = 12
    assert geopandas.options.display_precision == 12

    # The following assignment actually raises ValueError
    with pytest.raises(ValueError):
        geopandas.options.display_precision = 3.0

    # Test with invalid (non-numeric) string, should raise ValueError
    with pytest.raises(ValueError):
        geopandas.options.display_precision = ""

    # Test with negative float
    with pytest.raises(ValueError):
        geopandas.options.display_precision = -2.5

    geopandas.options.display_precision = None


def test_options_io_engine():
    # Edge case: default should be None
    assert geopandas.options.io_engine is None

    # Test with all supported values and unusual case
    geopandas.options.io_engine = "fiona"
    assert geopandas.options.io_engine == "fiona"

    geopandas.options.io_engine = "pyogrio"
    assert geopandas.options.io_engine == "pyogrio"

    # Empty string should raise ValueError
    with pytest.raises(ValueError):
        geopandas.options.io_engine = ""

    # Try a numeric value: zero
    with pytest.raises(ValueError):
        geopandas.options.io_engine = 0

    geopandas.options.io_engine = None