import pytest
from _plotly_utils.basevalidators import AnyValidator
import numpy as np
from ...test_optional.test_utils.test_utils import np_nan, np_inf


# Fixtures


@pytest.fixture()
def validator():
    return AnyValidator("prop", "parent")


@pytest.fixture()
def validator_aok():
    return AnyValidator("prop", "parent", array_ok=True)


# Tests


# Acceptance
@pytest.mark.parametrize(
    "val", [
        set(), 
        "Hello", 
        123, 
        np_inf(), 
        np_nan(), 
        {},
        "",                  # empty string edge case
        0,                   # zero edge case
        -9999999999,         # large negative integer
        1e100,               # large positive float
        [1, 2, 3],           # list input
        (4, 5, 6),           # tuple input
        [np_nan(), np_inf()],# list with np_nan and np_inf
        None,                # None edge case
    ]
)
def test_acceptance(val, validator):
    assert validator.validate_coerce(val) is val


# Acceptance of arrays
@pytest.mark.parametrize(
    "val",
    [
        0,                              # zero as scalar
        -123456789,                     # large negative int
        1.23456789e9,                   # large float
        "",                             # empty string
        [np_nan(), np_inf(), 42, ""],   # list with mixed edge cases
        np.array([0, -1, 1, np_nan(), np_inf()]), # numpy array with edge cases
        ['a', '', 'b', 'c'],            # list of strings with empty string
        (),                             # empty tuple
        np.array([]),                   # empty numpy array
        ([{"x": None}, {"y": []}],),    # single-element tuple containing dicts
        ([1, 2.2, 3], [4, 5.5, 6]),     # tuple of lists with ints and floats
        ["Hello", 0, {}, None],         # list with string, zero, dict, None
        [np.pi, np.e, {}, None],        # list with float, dict, None
    ],
)
def test_acceptance_array(val, validator_aok):
    coerce_val = validator_aok.validate_coerce(val)
    if isinstance(val, np.ndarray):
        assert isinstance(coerce_val, np.ndarray)
        assert coerce_val.dtype == "object"
        # For arrays with mixed types, instead of np.array_equal (which fails on string/object dtypes), compare element-wise
        # FIX: Compare as floats with nan!=nan case handled
        for a, b in zip(coerce_val, val):
            if isinstance(a, float) and np.isnan(a) and isinstance(b, float) and np.isnan(b):
                continue
            assert a == b
    elif isinstance(val, (list, tuple)):
        assert coerce_val == list(val)
        assert validator_aok.present(coerce_val) == tuple(val)
    else:
        assert coerce_val == val
        assert validator_aok.present(coerce_val) == val