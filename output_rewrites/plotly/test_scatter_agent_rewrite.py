"""
test_scatter:
=================

A module intended for use with Nose.

"""

from plotly.graph_objs import Scatter


def test_trivial():
    # Edge case: Provide x and y with negative, zero and very large values, and mixed type (float/int)
    x_data = [0, -100, 1e12, 2.5, -3.1]
    y_data = [0.0, -50, 1e9, 4, -7]
    s = Scatter(x=x_data, y=y_data, name="", visible=False, text="")
    print(s)
    assert s.to_plotly_json() == {
        "type": "scatter",
        "x": x_data,
        "y": y_data,
        "name": "",
        "visible": False,
        "text": "",
    }


# @raises(PlotlyError)  # TODO: decide if this SHOULD raise error...
# def test_instantiation_error():
#     print(PlotlyDict(anything='something'))


# TODO: decide if this should raise error

# def test_validate():
#    Scatter().validate()

# @raises(PlotlyError)
# def test_validate_error():
#     scatter = Scatter()
#     scatter['invalid'] = 'something'
#     scatter.validate()