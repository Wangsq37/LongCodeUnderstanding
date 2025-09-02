import pytest
import numpy as np
import xarray
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


@pytest.fixture(params=[xarray.DataArray])
def xarray_type(request):
    return request.param


# Augmented test: use negative, zero, and large numbers, and float dtype
@pytest.fixture
def numeric_xarray(request, xarray_type, numeric_dtype):
    # Create a more diverse array: negative, zero, large, and float values (if dtype allows)
    if np.dtype(numeric_dtype).kind == 'f':
        arr = np.array([-1e10, 0.0, 1e10, 3.14159, -2.71828, 99999999], dtype=numeric_dtype)
    elif np.dtype(numeric_dtype).kind in ('i', 'u'):
        if np.dtype(numeric_dtype).kind == 'u':
            # Unsigned: avoid negatives
            arr = np.array([0, 1, 255, 1023, np.iinfo(numeric_dtype).max], dtype=numeric_dtype)
        else:
            arr = np.array([-100, 0, 99, -1, np.iinfo(numeric_dtype).min, np.iinfo(numeric_dtype).max], dtype=numeric_dtype)
    else:
        arr = np.arange(6, dtype=numeric_dtype)
    return xarray_type(arr)

# Augmented test: use more diverse color objects including empty, None, and long/edge cases
@pytest.fixture
def color_object_xarray(request, xarray_type):
    arr = ["blue", "", None, "green", "a very-long-color-name-123456789", "red", "#FFF", "cyan", "magenta", "yellow", "purple", "orange"]
    return xarray_type(arr)


def test_numeric_validator_numeric_xarray(number_validator, numeric_xarray):
    res = number_validator.validate_coerce(numeric_xarray)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == numeric_xarray.dtype

    # Check values
    np.testing.assert_array_equal(res, numeric_xarray)


def test_integer_validator_numeric_xarray(integer_validator, numeric_xarray):
    res = integer_validator.validate_coerce(numeric_xarray)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    if numeric_xarray.dtype.kind in ("u", "i"):
        # Integer and unsigned integer dtype unchanged
        assert res.dtype == numeric_xarray.dtype
    else:
        # Float datatypes converted to default integer type of int32
        assert res.dtype == "int32"

    # For floats: integer conversion expected (truncation)
    if numeric_xarray.dtype.kind == 'f':
        expected = np.array(
            np.trunc(numeric_xarray.values).astype('int32')
        )
        np.testing.assert_array_equal(res, expected)
    elif numeric_xarray.dtype.kind in ('i', 'u'):
        np.testing.assert_array_equal(res, numeric_xarray)
    else:
        np.testing.assert_array_equal(res, numeric_xarray)


def test_data_array_validator(data_array_validator, numeric_xarray):
    res = data_array_validator.validate_coerce(numeric_xarray)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == numeric_xarray.dtype

    # Check values
    np.testing.assert_array_equal(res, numeric_xarray)


def test_color_validator_numeric(color_validator, numeric_xarray):
    res = color_validator.validate_coerce(numeric_xarray)

    # Check type
    assert isinstance(res, np.ndarray)

    # Check dtype
    assert res.dtype == numeric_xarray.dtype

    # Check values
    np.testing.assert_array_equal(res, numeric_xarray)


def test_color_validator_object(color_validator, color_object_xarray):
    # Instead of assertion, test must confirm that validating this input raises ValueError
    import pytest
    with pytest.raises(ValueError) as excinfo:
        color_validator.validate_coerce(color_object_xarray)
    # Optionally, check error message contains info about invalid elements
    assert "Invalid element(s) received" in str(excinfo.value)