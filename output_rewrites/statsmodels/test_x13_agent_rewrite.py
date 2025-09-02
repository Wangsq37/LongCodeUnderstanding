import pandas as pd

from statsmodels.tsa.x13 import _make_var_names


def test_make_var_names():
    # Edge case: Series with empty string as name
    exog = pd.Series([0, -1, 999999], name="")
    assert _make_var_names(exog) == ""

    # Edge case: Series with name as None (should possibly return None or handle gracefully)
    exog_none = pd.Series([-42, 42, 3.14], name=None)
    assert _make_var_names(exog_none) == "x1"

    # Large numbers, float name
    exog_floatname = pd.Series([1e10, 5e-5, -2], name="variable_ΔΣ")
    assert _make_var_names(exog_floatname) == "variable_ΔΣ"

    # Series with numeric (int) as name
    exog_intname = pd.Series([10, 20, 30], name=1234)
    assert _make_var_names(exog_intname) == "x1"

    # Series with negative name (unlikely, but test)
    exog_negname = pd.Series([100, 200], name=-99)
    assert _make_var_names(exog_negname) == "x1"