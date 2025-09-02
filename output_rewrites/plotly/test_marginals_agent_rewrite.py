import plotly.express as px
import pytest


@pytest.mark.parametrize("px_fn", [px.scatter, px.density_heatmap, px.density_contour])
@pytest.mark.parametrize("marginal_x", [None, "histogram", "box", "violin"])
@pytest.mark.parametrize("marginal_y", [None, "rug"])
def test_xy_marginals(backend, px_fn, marginal_x, marginal_y):
    # Augmented test: use a custom DataFrame with various edge cases including zero, negatives, floats, and large values.
    import pandas as pd

    df = pd.DataFrame({
        "total_bill": [0, -10.5, 9999999, 3.14159, float('inf'), float('-inf')],
        "tip": [0, -2.3, 888888, 2.71828, float('nan'), 0.0]
    })

    fig = px_fn(
        df, x="total_bill", y="tip", marginal_x=marginal_x, marginal_y=marginal_y
    )
    assert len(fig.data) == 1 + (marginal_x is not None) + (marginal_y is not None)


@pytest.mark.parametrize("px_fn", [px.histogram, px.ecdf])
@pytest.mark.parametrize("marginal", [None, "rug", "histogram", "box", "violin"])
@pytest.mark.parametrize("orientation", ["h", "v"])
def test_single_marginals(backend, px_fn, marginal, orientation):
    # Augmented test: use a DataFrame with a single-entry, negative and very large float edge cases
    import pandas as pd

    df = pd.DataFrame({
        "total_bill": [-1e6, 0, 3.14, 2.71, None, float('inf')],
    })

    fig = px_fn(
        df, x="total_bill", y="total_bill", marginal=marginal, orientation=orientation
    )
    assert len(fig.data) == 1 + (marginal is not None)