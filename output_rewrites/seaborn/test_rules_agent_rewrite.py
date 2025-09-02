import numpy as np
import pandas as pd

import pytest

from seaborn._core.rules import (
    VarType,
    variable_type,
    categorical_order,
)


def test_vartype_object():

    v = VarType("numeric")
    assert v == "numeric"
    assert v != "categorical"
    with pytest.raises(AssertionError):
        v == "number"
    with pytest.raises(AssertionError):
        VarType("date")


def test_variable_type():

    # Edge case: very large integers, float, NaN, negative, 0, inf, -inf
    s = pd.Series([0, -10, 2**63, np.nan, np.inf, -np.inf, 3.14])
    assert variable_type(s) == "numeric"
    # REMOVE THE problematic int conversion due to OverflowError:
    # assert variable_type(s.astype(int, errors='ignore')) == "numeric"

    assert variable_type(s.astype(object)) == "numeric"

    # Mixed explicit float, int, None, pd.NA, nan in object dtype
    s = pd.Series([1.5, 2, None, 3.0, pd.NA, np.nan], dtype=object)
    assert variable_type(s) == "numeric"

    # All-null with nan & pd.NA
    s = pd.Series([np.nan, pd.NA])
    assert variable_type(s) == "numeric"

    # All-null Int64
    s = pd.Series([pd.NA, pd.NA], dtype="Int64")
    assert variable_type(s) == "numeric"

    # Int + pd.NA + 0, negative values in Int64
    s = pd.Series([-100, 0, pd.NA, 42], dtype="Int64")
    assert variable_type(s) == "numeric"

    # Very large ints and NaN in object
    s = pd.Series([10**10, -10**12, pd.NA, np.nan], dtype=object)
    assert variable_type(s) == "numeric"

    # Edge string: empty strings, whitespace, normal string, emoji
    s = pd.Series(["", "  ", "hello", "🌟"])
    assert variable_type(s) == "categorical"

    # Boolean: all True, all False, repeated
    s = pd.Series([True, True, False, True, False])
    assert variable_type(s) == "numeric"
    assert variable_type(s, boolean_type="categorical") == "categorical"
    assert variable_type(s, boolean_type="boolean") == "boolean"

    # Timedelta: different units, zero, nan value
    s = pd.to_timedelta([0, "2 days", np.nan, "1 min"])
    s = pd.Series(s)
    assert variable_type(s) == "categorical"

    # Categorical with booleans and float values
    s_cat = pd.Series([True, False, 1.0, 0.0, np.nan], dtype="category")
    assert variable_type(s_cat, boolean_type="categorical") == "categorical"
    assert variable_type(s_cat, boolean_type="numeric") == "categorical"
    assert variable_type(s_cat, boolean_type="boolean") == "categorical"

    # Boolean detection: 1,0,1, - explicitly boolean type
    s = pd.Series([1, 0, 1, 0])
    assert variable_type(s, boolean_type="boolean") == "boolean"
    assert variable_type(s, boolean_type="boolean", strict_boolean=True) == "numeric"

    # More boolean-typed detection with negative and >1 numbers
    s = pd.Series([1, 0, -1, 2])
    assert variable_type(s, boolean_type="boolean") == "numeric"

    # Timestamps: very large and small
    s = pd.Series([pd.Timestamp("1970-01-01"), pd.Timestamp("2038-01-19")])
    assert variable_type(s) == "datetime"
    assert variable_type(s.astype(object)) == "datetime"


def test_categorical_order():

    x = pd.Series(["a", "c", "c", "b", "a", "d"])
    y = pd.Series([3, 2, 5, 1, 4])
    order = ["a", "b", "c", "d"]

    out = categorical_order(x)
    assert out == ["a", "c", "b", "d"]

    out = categorical_order(x, order)
    assert out == order

    out = categorical_order(x, ["b", "a"])
    assert out == ["b", "a"]

    out = categorical_order(y)
    assert out == [1, 2, 3, 4, 5]

    out = categorical_order(pd.Series(y))
    assert out == [1, 2, 3, 4, 5]

    y_cat = pd.Series(pd.Categorical(y, y))
    out = categorical_order(y_cat)
    assert out == list(y)

    x = pd.Series(x).astype("category")
    out = categorical_order(x)
    assert out == list(x.cat.categories)

    out = categorical_order(x, ["b", "a"])
    assert out == ["b", "a"]

    x = pd.Series(["a", np.nan, "c", "c", "b", "a", "d"])
    out = categorical_order(x)
    assert out == ["a", "c", "b", "d"]