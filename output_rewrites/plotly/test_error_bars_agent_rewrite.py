"""
test_error_bars:
================

A module intended for use with Nose.

"""

from plotly.graph_objs import ErrorX, ErrorY


def test_instantiate_error_x():
    ErrorX()
    ErrorX(
        array=[1, 2, 3],
        arrayminus=[2, 1, 2],
        color="red",
        copy_ystyle=False,
        symmetric=False,
        thickness=2,
        type="percent",
        value=1,
        valueminus=4,
        visible=True,
        width=5,
    )


def test_instantiate_error_y():
    ErrorY()
    ErrorY(
        array=[1, 2, 3],
        arrayminus=[2, 1, 2],
        color="red",
        symmetric=False,
        thickness=2,
        type="percent",
        value=1,
        valueminus=4,
        visible=True,
        width=5,
    )


def test_key_error():
    # Augmented test case with more diverse and edge-case-like data
    assert ErrorX(value=0, typ="", color=None, array=[-1, 0, 1e6]) == {
        "color": None,
        "typ": "",
        "value": 0,
        "array": [-1, 0, 1e6],
    }