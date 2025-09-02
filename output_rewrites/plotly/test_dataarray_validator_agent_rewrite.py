import pytest
from _plotly_utils.basevalidators import DataArrayValidator
import numpy as np
import pandas as pd


# Fixtures
@pytest.fixture()
def validator():
    return DataArrayValidator("prop", "parent")


# Tests


# Acceptance
@pytest.mark.parametrize(
    "val",
    [
        [0, -1, 999999999],
        [3.1415, -2.718, 0.0],
        [None, "foo", "", 42],
        tuple(),
        ("", None, True, False),
        ["X", -99.9, 0, "Y", np.nan],
        [np.array(-7), np.array(0), np.array(1.618)],
    ],
)
def test_validator_acceptance_simple(val, validator):
    coerce_val = validator.validate_coerce(val)
    assert isinstance(coerce_val, list)
    assert validator.present(coerce_val) == tuple(val)


@pytest.mark.parametrize(
    "val",
    [np.array([2, 3, 4]), pd.Series(["a", "b", "c"]), np.array([[1, 2, 3], [4, 5, 6]])],
)
def test_validator_acceptance_homogeneous(val, validator):
    coerce_val = validator.validate_coerce(val)
    assert isinstance(coerce_val, np.ndarray)
    assert np.array_equal(validator.present(coerce_val), val)


# Rejection
@pytest.mark.parametrize("val", ["Hello", 23, set(), {}])
def test_rejection(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)