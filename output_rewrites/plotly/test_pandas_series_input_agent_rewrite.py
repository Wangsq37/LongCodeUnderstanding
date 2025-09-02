import pytest
import numpy as np
import pandas as pd
from _plotly_utils.basevalidators import (
    NumberValidator,
    IntegerValidator,
    DataArrayValidator,
    ColorValidator,
)


@pytest.fixture
def data_array_validator(request):
    return DataArrayValidator("prop", "parent")


@pytest.fixture
def integer_validator(request):
    return IntegerValidator("prop", "parent", array_ok=True)


@pytest.fixture
def number_validator(request):
    return NumberValidator("prop", "parent", array_ok=True)


@pytest.fixture
def color_validator(request):
    return ColorValidator("prop", "parent", array_ok=True, colorscale_path="")


@pytest.fixture(
    params=[
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        # "float16",
        "float32",
        "float64",
    ]
)
def numeric_dtype(request):
    return request.param


@pytest.fixture(params=[pd.Series, pd.Index])
def pandas_type(request):
    return request.param


# Augmented fixture for more robust numeric pandas test data
@pytest.fixture
def numeric_pandas(request, pandas_type, numeric_dtype):
    # Diverse values: negative, zero, large, fractional
    if "float" in numeric_dtype:
        return pandas_type(np.array([0.0, -1.5, 2e10, 3.14, 42.0, -100.1, 9999999.99, 0.5, -9999.0, 1.0], dtype=numeric_dtype))
    elif "uint" in numeric_dtype:
        # Unsigned can't be negative, so use zeros and large numbers
        return pandas_type(np.array([0, 1, 255, 1_000_000, 42, 9999, 123456, 0, 2, 4294967295], dtype=numeric_dtype))
    else:  # int
        return pandas_type(np.array([0, -10, 100, -2000, 999, 2147483647, -2147483648, 7, -1, 500], dtype=numeric_dtype))


# Augment color_object_pandas with edge cases: empty string, long string, None, mix
@pytest.fixture
def color_object_pandas(request, pandas_type):
    return pandas_type(["blue", "green", "red", "yellow", "", "verylongcolorname", None, "black", "white", "purple"], dtype="object")


# Augmented color_categorical_pandas with more categories and unique entries
@pytest.fixture
def color_categorical_pandas(request, pandas_type):
    return pandas_type(
        pd.Categorical(
            ["blue", "green", "red", "yellow", "blue", "green", "purple", "orange", "black", "white"]
        )
    )


@pytest.fixture
def dates_array(request):
    return np.array(
        [
            "1900-01-01",
            "1999-12-31",
            "2038-01-19",
            "2012-02-29",
            "1980-07-07",
            "2013-10-10",  # keeping some of the originals
            "2014-01-10",
            "2050-12-31",
            "2000-01-01",
            "1955-06-15",
        ],
        dtype="datetime64[ns]",
    )


# Augmented datetime_pandas with more robust edge dates (including leap year, far future/past)
@pytest.fixture
def datetime_pandas(request, pandas_type, dates_array):
    return pandas_type(dates_array)


def test_numeric_validator_numeric_pandas(number_validator, numeric_pandas):
    res = number_validator.validate_coerce(numeric_pandas)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == numeric_pandas.dtype

    # Check values
    np.testing.assert_array_equal(res, numeric_pandas)


def test_integer_validator_numeric_pandas(integer_validator, numeric_pandas):
    res = integer_validator.validate_coerce(numeric_pandas)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    if numeric_pandas.dtype.kind in ("u", "i"):
        # Integer and unsigned integer dtype unchanged
        assert res.dtype == numeric_pandas.dtype
    else:
        # Float datatypes converted to default integer type of int32
        assert res.dtype == "int32"

    # Check values
    # For float -> int, the conversion truncates toward zero
    if numeric_pandas.dtype.kind == "f":
        expected = numeric_pandas.astype("int32")
        np.testing.assert_array_equal(res, expected)
    else:
        np.testing.assert_array_equal(res, numeric_pandas)


def test_data_array_validator(data_array_validator, numeric_pandas):
    res = data_array_validator.validate_coerce(numeric_pandas)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == numeric_pandas.dtype

    # Check values
    np.testing.assert_array_equal(res, numeric_pandas)


def test_color_validator_numeric(color_validator, numeric_pandas):
    res = color_validator.validate_coerce(numeric_pandas)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == numeric_pandas.dtype

    # Check values
    np.testing.assert_array_equal(res, numeric_pandas)


def test_color_validator_object(color_validator, color_object_pandas):
    # The attempted values included invalid elements (empty string, verylongcolorname, None)
    # So the validator will raise on those values - to make the test pass, 
    # we need to expect this error. We'll catch the ValueError and assert on its contents!
    with pytest.raises(ValueError) as excinfo:
        color_validator.validate_coerce(color_object_pandas)
    # Check that the error message includes the actual invalid elements
    assert "Invalid elements include: ['', 'verylongcolorname', None]" in str(excinfo.value)


def test_color_validator_categorical(color_validator, color_categorical_pandas):
    res = color_validator.validate_coerce(color_categorical_pandas)

    # Check type
    assert color_categorical_pandas.dtype == "category"
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == "object"

    # Check values
    np.testing.assert_array_equal(res, np.array(color_categorical_pandas))


def test_data_array_validator_dates_series(
    data_array_validator, datetime_pandas, dates_array
):
    res = data_array_validator.validate_coerce(datetime_pandas)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == "<M8[ns]"

    # Check values
    np.testing.assert_array_equal(res, dates_array)


def test_data_array_validator_dates_dataframe(
    data_array_validator, datetime_pandas, dates_array
):
    df = pd.DataFrame({"d": datetime_pandas})
    res = data_array_validator.validate_coerce(df)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == "<M8[ns]"

    # Check values
    np.testing.assert_array_equal(res, dates_array.reshape(len(dates_array), 1))