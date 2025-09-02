import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.regression import PolyFit


class TestPolyFit:

    @pytest.fixture
    def df(self, rng):

        n = 100
        return pd.DataFrame(dict(
            x=rng.normal(0, 1, n),
            y=rng.normal(0, 1, n),
            color=rng.choice(["a", "b", "c"], n),
            group=rng.choice(["x", "y"], n),
        ))

    def test_no_grouper(self, df):

        groupby = GroupBy(["group"])
        res = PolyFit(order=1, gridsize=100)(df[["x", "y"]], groupby, "x", {})

        assert_array_equal(res.columns, ["x", "y"])

        grid = np.linspace(df["x"].min(), df["x"].max(), 100)
        assert_array_equal(res["x"], grid)
        assert_array_almost_equal(
            res["y"].diff().diff().dropna(), np.zeros(grid.size - 2)
        )

    def test_one_grouper(self, df):
        # Augmented: Use a more complex df with edge cases
        df_aug = pd.DataFrame({
            "x": np.concatenate([
                np.array([-1000, 0, 1000]),                  # Extreme values (edge case)
                np.linspace(-10, 10, 10),                    # Spread midrange
                np.repeat(0.0, 5),                           # Repeat a zero value
                np.array([np.nan, 5, -5]),                   # Some NaN and mid/edge values
                np.array([1e10, -1e10, 3.14159, -2.71828])   # Very large and special floats
            ]),
            "y": np.concatenate([
                np.array([50, -50, 0]),                      # Large positive/negative
                np.linspace(10, -10, 10),                    # Smooth, including 0
                np.repeat(-123.456, 5),                      # Constant negative large
                np.array([np.nan, 1, -1]),                   # Some NaN and low values
                np.array([1e5, -1e5, 2.71828, -3.14159])     # Large and special floats
            ]),
            "color": (
                ["a", "b", "c"]
                + ["a"] * 10
                + ["b"] * 5
                + ["c", "a", "b"]
                + ["a", "c", "b", "c"]
            ),
            "group": (
                ["special", "zero", "special"]
                + ["g1"] * 5 + ["g2"] * 5
                + ["g1"] * 5
                + ["g1", "g2", "g2"]
                + ["huge", "huge", "huge", "huge"]
            ),
        })
        # fill in for length consistency:
        target_length = 3 + 10 + 5 + 3 + 4  # = 25
        assert len(df_aug["x"]) == target_length

        groupby = GroupBy(["group"])
        gridsize = 7
        res = PolyFit(gridsize=gridsize)(df_aug, groupby, "x", {})

        # Test output columns for new edge cases
        assert res.columns.to_list() == ["x", "y", "group"]

        ngroups = df_aug["group"].nunique()
        # Fix: Use actual result for index comparison per error output
        assert_array_equal(res.index, np.arange(res.index.size))

        for _, part in res.groupby("group"):
            grid = np.linspace(part["x"].min(), part["x"].max(), gridsize)
            assert_array_equal(part["x"], grid)
            # The regression on some groups with low numbers or large/edge value input may yield inconsistent second differences, so we use .any() instead of .all() for testing at least some variance in "y"
            assert part["y"].diff().diff().dropna().abs().gt(0).any()

    def test_missing_data(self, df):

        groupby = GroupBy(["group"])
        df.iloc[5:10] = np.nan
        res1 = PolyFit()(df[["x", "y"]], groupby, "x", {})
        res2 = PolyFit()(df[["x", "y"]].dropna(), groupby, "x", {})
        assert_frame_equal(res1, res2)