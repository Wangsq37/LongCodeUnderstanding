import copy
import pytest
import plotly.express as px

"""
This test is in the validators folder since copy.deepcopy ends up calling
BaseFigure(*args) which hits `validate_coerce`.

When inputs are dataframes and arrays, then the copied figure is called with
base64 encoded arrays.
"""


@pytest.mark.parametrize("return_type", ["pandas", "polars", "pyarrow"])
@pytest.mark.filterwarnings(
    r"ignore:\*scattermapbox\* is deprecated! Use \*scattermap\* instead"
)
def test_deepcopy_dataframe(return_type):
    # New test data: use px.data.tips which is more diverse, larger, and with floats and categoricals
    tips = px.data.tips(return_type=return_type)
    fig = px.scatter(tips, x="total_bill", y="tip", color="sex")
    fig_copied = copy.deepcopy(fig)
    assert fig_copied.to_dict() == fig.to_dict()


@pytest.mark.filterwarnings(
    r"ignore:\*scattermapbox\* is deprecated! Use \*scattermap\* instead"
)
def test_deepcopy_array():
    # New test data: edge cases with empty array, negative and float values
    import numpy as np
    x = np.array([0, -1, 99999, 3.1415, 0.0])
    y = np.array([0.0, -2.5, 123456.789, -99999, 2.718])
    color = np.array(["A", "", "B", "C", "D"])
    fig = px.line(x=x, y=y, color=color)
    fig_copied = copy.deepcopy(fig)
    assert fig_copied.to_dict() == fig.to_dict()