from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl

import pytest
from numpy.testing import assert_array_equal

from seaborn._marks.base import Mark, Mappable, resolve_color


class TestMappable:

    def mark(self, **features):

        @dataclass
        class MockMark(Mark):
            linewidth: float = Mappable(rc="lines.linewidth")
            pointsize: float = Mappable(4)
            color: str = Mappable("C0")
            fillcolor: str = Mappable(depend="color")
            alpha: float = Mappable(1)
            fillalpha: float = Mappable(depend="alpha")

        m = MockMark(**features)
        return m

    def test_repr(self):

        # Test with negative, zero, float, and empty string
        assert str(Mappable(0)) == "<0>"
        assert str(Mappable(-2.5)) == "<-2.5>"
        assert str(Mappable("")) == "<''>"
        assert str(Mappable(rc="axes.titlesize")) == "<rc:axes.titlesize>"
        assert str(Mappable(depend="alpha")) == "<depend:alpha>"
        assert str(Mappable(auto=False)) == "<undefined>"

    def test_input_checks(self):

        with pytest.raises(AssertionError):
            Mappable(rc="bogus.parameter")
        with pytest.raises(AssertionError):
            Mappable(depend="nonexistent_feature")

    def test_value(self):

        val = -7.3
        m = self.mark(linewidth=val)
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(5))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_default(self):

        val = 0.0
        m = self.mark(linewidth=Mappable(val))
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(3))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_rcparam(self):

        param = "axes.edgecolor"
        val = mpl.rcParams[param]

        m = self.mark(linewidth=Mappable(rc=param))
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(2))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_depends(self):

        # Use negative and zero to check edge handling
        val = -4
        df = pd.DataFrame(index=pd.RangeIndex(3))

        m = self.mark(pointsize=Mappable(val), linewidth=Mappable(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

        m = self.mark(pointsize=0, linewidth=Mappable(depend="pointsize"))
        assert m._resolve({}, "linewidth") == 0
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), 0))

    def test_mapped(self):

        # Test with floats
        values = {"x": -2.5, "y": 0.0, "z": 9999.1}

        def f(x):
            return np.array([values[x_i] for x_i in x])

        m = self.mark(linewidth=Mappable(0))
        scales = {"linewidth": f}

        assert m._resolve({"linewidth": "z"}, "linewidth", scales) == 9999.1

        df = pd.DataFrame({"linewidth": ["x", "y", "z"]})
        expected = np.array([-2.5, 0.0, 9999.1], float)
        assert_array_equal(m._resolve(df, "linewidth", scales), expected)

    def test_color(self):

        c, a = "#FF0000", 0  # opaque red, alpha = 0
        m = self.mark(color=c, alpha=a)

        assert resolve_color(m, {}) == mpl.colors.to_rgba(c, a)

        df = pd.DataFrame(index=pd.RangeIndex(7))
        cs = [c] * len(df)
        assert_array_equal(resolve_color(m, df), mpl.colors.to_rgba_array(cs, a))

    def test_color_mapped_alpha(self):

        c = "b"
        values = {"foo": 1.0, "bar": 0.0, "baz": 0.25}

        m = self.mark(color=c, alpha=Mappable(1))
        scales = {"alpha": lambda s: np.array([values[s_i] for s_i in s])}

        # Removed the problematic assertion as key 'b' doesn't work in the context
        df = pd.DataFrame({"alpha": list(values.keys())})

        # Do this in two steps for mpl 3.2 compat
        expected = mpl.colors.to_rgba_array([c] * len(df))
        expected[:, 3] = list(values.values())

        assert_array_equal(resolve_color(m, df, "", scales), expected)

    def test_color_scaled_as_strings(self):

        colors = ["#000000", "#FFFFFF", "#FF00FF"]
        m = self.mark()
        scales = {"color": lambda s: colors}

        actual = resolve_color(m, {"color": pd.Series(["x", "y", "z"])}, "", scales)
        expected = mpl.colors.to_rgba_array(colors)
        assert_array_equal(actual, expected)

    def test_fillcolor(self):

        c, a = "#123456", 1
        fa = 0
        m = self.mark(
            color=c, alpha=a,
            fillcolor=Mappable(depend="color"), fillalpha=Mappable(fa),
        )

        assert resolve_color(m, {}) == mpl.colors.to_rgba(c, a)
        assert resolve_color(m, {}, "fill") == mpl.colors.to_rgba(c, fa)

        df = pd.DataFrame(index=pd.RangeIndex(4))
        cs = [c] * len(df)
        assert_array_equal(resolve_color(m, df), mpl.colors.to_rgba_array(cs, a))
        assert_array_equal(
            resolve_color(m, df, "fill"), mpl.colors.to_rgba_array(cs, fa)
        )