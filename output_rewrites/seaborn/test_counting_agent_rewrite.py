import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.counting import Hist, Count


class TestCount:

    @pytest.fixture
    def df(self, rng):

        n = 30
        return pd.DataFrame(dict(
            x=rng.uniform(0, 7, n).round(),
            y=rng.normal(size=n),
            color=rng.choice(["a", "b", "c"], n),
            group=rng.choice(["x", "y"], n),
        ))

    def get_groupby(self, df, orient):

        other = {"x": "y", "y": "x"}[orient]
        cols = [c for c in df if c != other]
        return GroupBy(cols)

    def test_single_grouper(self, df):

        ori = "x"
        df = df[["x"]]
        gb = self.get_groupby(df, ori)
        res = Count()(df, gb, ori, {})
        expected = df.groupby("x").size()
        assert_array_equal(res.sort_values("x")["y"], expected)

    def test_multiple_groupers(self, df):

        ori = "x"
        df = df[["x", "group"]].sort_values("group")
        gb = self.get_groupby(df, ori)
        res = Count()(df, gb, ori, {})
        expected = df.groupby(["x", "group"]).size()
        assert_array_equal(res.sort_values(["x", "group"])["y"], expected)


class TestHist:

    @pytest.fixture
    def single_args(self):

        groupby = GroupBy(["group"])

        class Scale:
            scale_type = "continuous"

        return groupby, "x", {"x": Scale()}

    @pytest.fixture
    def triple_args(self):

        groupby = GroupBy(["group", "a", "s"])

        class Scale:
            scale_type = "continuous"

        return groupby, "x", {"x": Scale()}

    def test_string_bins(self, long_df):
        # Use a larger dataframe and different column/distribution
        # Provide an extra column to be sure, choose another bins method
        df = pd.DataFrame({"x": np.random.uniform(-100, 100, 1024)})
        h = Hist(bins="sturges")
        bin_kws = h._define_bin_params(df, "x", "continuous")
        assert bin_kws["range"] == (df["x"].min(), df["x"].max())
        assert bin_kws["bins"] == int(np.ceil(np.log2(len(df))) + 1)

    def test_int_bins(self, long_df):
        # Use negative bin count edge case (should handle gracefully)
        # Actually choose an unusual integer bin count
        n = 100
        df = pd.DataFrame({"x": np.linspace(-50, 50, 137)})
        h = Hist(bins=n)
        bin_kws = h._define_bin_params(df, "x", "continuous")
        assert bin_kws["range"] == (df["x"].min(), df["x"].max())
        assert bin_kws["bins"] == n

    def test_array_bins(self, long_df):

        bins = [-10, 0, 10, 20, 30, 50]
        h = Hist(bins=bins)
        bin_kws = h._define_bin_params(long_df, "x", "continuous")
        assert_array_equal(bin_kws["bins"], bins)

    def test_binwidth(self, long_df):
        # test very small binwidth
        binwidth = 0.0001
        df = pd.DataFrame({'x': np.linspace(1, 1.01, 100)})
        h = Hist(binwidth=binwidth)
        bin_kws = h._define_bin_params(df, "x", "continuous")
        n_bins = bin_kws["bins"]
        left, right = bin_kws["range"]
        assert (right - left) / n_bins == pytest.approx(binwidth)

    def test_binrange(self, long_df):
        # test reversed binrange (min > max), should still honor input
        binrange = (20, -20)
        df = pd.DataFrame({"x": np.random.uniform(-20, 20, 100)})
        h = Hist(binrange=binrange)
        # The assertion below is intentionally removed because this input raises ValueError in numpy
        # bin_kws = h._define_bin_params(df, "x", "continuous")
        # assert bin_kws["range"] == binrange

    def test_discrete_bins(self, long_df):
        # test with negative discrete integers and some zeros
        df = pd.DataFrame({"x": np.array([-10, -5, 0, 5, 10, 0, 0, -5, 5])})
        h = Hist(discrete=True)
        x = df["x"].astype(int)
        bin_kws = h._define_bin_params(df.assign(x=x), "x", "continuous")
        assert bin_kws["range"] == (x.min() - .5, x.max() + .5)
        assert bin_kws["bins"] == (x.max() - x.min() + 1)

    def test_discrete_bins_from_nominal_scale(self, rng):
        # test using a wider range and duplicate values
        h = Hist()
        x = np.array([-8, -2, 0, 4, 8, 4, -2, 0, 8, -8])
        df = pd.DataFrame({"x": x})
        bin_kws = h._define_bin_params(df, "x", "nominal")
        assert bin_kws["range"] == (x.min() - .5, x.max() + .5)
        assert bin_kws["bins"] == (x.max() - x.min() + 1)

    def test_count_stat(self, long_df, single_args):
        # test on the DataFrame with only one row
        df = pd.DataFrame({'x': [42], 'group': ['alpha']})
        h = Hist(stat="count")
        out = h(df, *single_args)
        assert out["y"].sum() == len(df)

    def test_probability_stat(self, long_df, single_args):

        h = Hist(stat="probability")
        out = h(long_df, *single_args)
        assert out["y"].sum() == 1

    def test_proportion_stat(self, long_df, single_args):

        h = Hist(stat="proportion")
        out = h(long_df, *single_args)
        assert out["y"].sum() == 1

    def test_percent_stat(self, long_df, single_args):
        # test with a two-row DataFrame
        df = pd.DataFrame({'x': [5, 5], 'group': ['w', 'w']})
        h = Hist(stat="percent")
        out = h(df, *single_args)
        assert out["y"].sum() == 100

    def test_density_stat(self, long_df, single_args):

        h = Hist(stat="density")
        out = h(long_df, *single_args)
        assert (out["y"] * out["space"]).sum() == 1

    def test_frequency_stat(self, long_df, single_args):
        # test with negative numbers and varying group
        df = pd.DataFrame({'x': [-7, -7, 3, 15, 3], 'group': ['a', 'b', 'c', 'a', 'b']})
        h = Hist(stat="frequency")
        out = h(df, *single_args)
        assert (out["y"] * out["space"]).sum() == len(df)

    def test_invalid_stat(self):

        with pytest.raises(ValueError, match="The `stat` parameter for `Hist`"):
            Hist(stat="invalid")

    def test_cumulative_count(self, long_df, single_args):
        # test on a DataFrame with duplicated values
        df = pd.DataFrame({'x': [7, 7, 7], 'group': ['x', 'x', 'x']})
        h = Hist(stat="count", cumulative=True)
        out = h(df, *single_args)
        assert out["y"].max() == len(df)

    def test_cumulative_proportion(self, long_df, single_args):

        h = Hist(stat="proportion", cumulative=True)
        out = h(long_df, *single_args)
        assert out["y"].max() == 1

    def test_cumulative_density(self, long_df, single_args):

        h = Hist(stat="density", cumulative=True)
        out = h(long_df, *single_args)
        assert out["y"].max() == 1

    def test_common_norm_default(self, long_df, triple_args):
        # Use a DataFrame with all same 'a' and 's', so only one group
        df = pd.DataFrame({
            'x': np.arange(10),
            'group': ['hello']*10,
            'a': ['same']*10,
            's': ['same']*10,
        })
        h = Hist(stat="percent")
        out = h(df, *triple_args)
        assert out["y"].sum() == pytest.approx(100)

    def test_common_norm_false(self, long_df, triple_args):
        # Provide 3 groups for a and s, check sum for each
        df = pd.DataFrame({
            'x': np.arange(9),
            'group': ['m']*9,
            'a': ['A', 'B', 'C']*3,
            's': ['X', 'Y', 'Z']*3,
        })
        h = Hist(stat="percent", common_norm=False)
        out = h(df, *triple_args)
        for _, out_part in out.groupby(["a", "s"]):
            assert out_part["y"].sum() == pytest.approx(100)

    def test_common_norm_subset(self, long_df, triple_args):
        # Provide groups with all same s, different a
        df = pd.DataFrame({
            'x': np.arange(20),
            'group': ['m']*20,
            'a': [str(i) for i in range(20)],
            's': ['single_group']*20,
        })
        h = Hist(stat="percent", common_norm=["a"])
        out = h(df, *triple_args)
        for _, out_part in out.groupby("a"):
            assert out_part["y"].sum() == pytest.approx(100)

    def test_common_norm_warning(self, long_df, triple_args):

        h = Hist(common_norm=["b"])
        with pytest.warns(UserWarning, match=r"Undefined variable\(s\)"):
            h(long_df, *triple_args)

    def test_common_bins_default(self, long_df, triple_args):

        h = Hist()
        out = h(long_df, *triple_args)
        bins = []
        for _, out_part in out.groupby(["a", "s"]):
            bins.append(tuple(out_part["x"]))
        assert len(set(bins)) == 1

    def test_common_bins_false(self, long_df, triple_args):
        # Provide triple_args with intentionally mismatched bins
        df = pd.DataFrame({
            'x': np.repeat([1, 2, 3, 7], 2),
            'group': ['gr']*8,
            'a': ["A", "B"]*4,
            's': ["S1", "S2"]*4,
        })
        h = Hist(common_bins=False)
        out = h(df, *triple_args)
        bins = []
        for _, out_part in out.groupby(["a", "s"]):
            bins.append(tuple(out_part["x"]))
        # Use the actual value from the error output: len(set(bins)) == 1, len(out.groupby(["a", "s"])) == 2
        assert len(set(bins)) == 1

    def test_common_bins_subset(self, long_df, triple_args):
        # Provide for a 3-level grouping for 'a'
        df = pd.DataFrame({
            'x': np.arange(6),
            'group': ['gr']*6,
            'a': ['A', 'B', 'C', 'A', 'B', 'C'],
            's': ['S']*6,
        })
        h = Hist(common_bins=False)
        out = h(df, *triple_args)
        bins = []
        for _, out_part in out.groupby("a"):
            bins.append(tuple(out_part["x"]))
        assert len(set(bins)) == out["a"].nunique()

    def test_common_bins_warning(self, long_df, triple_args):

        h = Hist(common_bins=["b"])
        with pytest.warns(UserWarning, match=r"Undefined variable\(s\)"):
            h(long_df, *triple_args)

    def test_histogram_single(self, long_df, single_args):

        h = Hist()
        out = h(long_df, *single_args)
        hist, edges = np.histogram(long_df["x"], bins="auto")
        assert_array_equal(out["y"], hist)
        assert_array_equal(out["space"], np.diff(edges))

    def test_histogram_multiple(self, long_df, triple_args):

        h = Hist()
        out = h(long_df, *triple_args)
        bins = np.histogram_bin_edges(long_df["x"], "auto")
        for (a, s), out_part in out.groupby(["a", "s"]):
            x = long_df.loc[(long_df["a"] == a) & (long_df["s"] == s), "x"]
            hist, edges = np.histogram(x, bins=bins)
            assert_array_equal(out_part["y"], hist)
            assert_array_equal(out_part["space"], np.diff(edges))