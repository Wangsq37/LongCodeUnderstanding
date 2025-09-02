import pytest
import copy
import pickle

from plotly.tools import make_subplots
import plotly.graph_objs as go


# fixtures
# --------
@pytest.fixture
def fig1(request):
    # Augmented: Use larger numbers, floats, empty string in layout title, additional trace type, negative values
    return go.Figure(
        data=[
            {"type": "scattergl", "marker": {"color": "red", "size": 1000}, "x": [0, -100, 1e6], "y": [2.5, -3.3, 0]},
            {
                "type": "parcoords",
                "dimensions": [
                    {"values": [0.0, 500.0, -10000.0]},
                    {"values": [1e6, 0.0, -1e-6]}
                ],
                "line": {"color": "black"},
            },
            {
                "type": "bar",
                "x": ["a", "b", "c"],
                "y": [0, 0, 0],
                "marker": {"color": "purple"}
            }
        ],
        layout={"title": ""},  # Empty string as title
    )


@pytest.fixture
def fig_subplots(request):
    # Augmented: 4 rows, 4 cols, traces with negative and float values, add bar traces up front
    fig = make_subplots(4, 4)
    fig.add_scatter(y=[0, -10, 1e6], row=1, col=1, mode="markers")
    fig.add_scatter(y=[-3.5, 0.0, 1], row=3, col=4, mode="lines+text")
    fig.add_bar(y=[0.0, 100.5, -5.5], row=4, col=2)
    fig.add_bar(y=[-9, 0, 9], row=2, col=3)
    return fig


# Deep copy
# ---------
def test_deepcopy_figure(fig1):
    fig_copied = copy.deepcopy(fig1)

    # Contents should be equal
    assert fig_copied.to_dict() == fig1.to_dict()

    # Identities should be distinct
    assert fig_copied is not fig1
    assert fig_copied.layout is not fig1.layout
    assert fig_copied.data is not fig1.data


def test_deepcopy_figure_subplots(fig_subplots):
    fig_copied = copy.deepcopy(fig_subplots)

    # Contents should be equal
    assert fig_copied.to_dict() == fig_subplots.to_dict()

    # Subplot metadata should be equal
    assert fig_subplots._grid_ref == fig_copied._grid_ref
    assert fig_subplots._grid_str == fig_copied._grid_str

    # Identities should be distinct
    assert fig_copied is not fig_subplots
    assert fig_copied.layout is not fig_subplots.layout
    assert fig_copied.data is not fig_subplots.data

    # Should be possible to add new trace to subplot location
    fig_subplots.add_bar(y=[-1000, 0, 1000], row=1, col=4)
    fig_copied.add_bar(y=[-1000, 0, 1000], row=1, col=4)

    # And contents should be still equal
    assert fig_copied.to_dict() == fig_subplots.to_dict()


def test_deepcopy_layout(fig1):
    copied_layout = copy.deepcopy(fig1.layout)

    # Contents should be equal
    assert copied_layout == fig1.layout

    # Identities should not
    assert copied_layout is not fig1.layout

    # Original layout should still have fig1 as parent
    assert fig1.layout.parent is fig1

    # Copied layout should have no parent
    assert copied_layout.parent is None


# Pickling
# --------
def test_pickle_figure_round_trip(fig1):
    fig_copied = pickle.loads(pickle.dumps(fig1))

    # Contents should be equal
    assert fig_copied.to_dict() == fig1.to_dict()


def test_pickle_figure_subplots_round_trip(fig_subplots):
    fig_copied = pickle.loads(pickle.dumps(fig_subplots))

    # Contents should be equal
    assert fig_copied.to_dict() == fig_subplots.to_dict()

    # Should be possible to add new trace to subplot location
    fig_subplots.add_bar(y=[float("inf"), float("-inf"), 0.0], row=4, col=4)
    fig_copied.add_bar(y=[float("inf"), float("-inf"), 0.0], row=4, col=4)

    # And contents should be still equal
    assert fig_copied.to_dict() == fig_subplots.to_dict()


def test_pickle_layout(fig1):
    copied_layout = pickle.loads(pickle.dumps(fig1.layout))

    # Contents should be equal
    assert copied_layout == fig1.layout

    # Identities should not
    assert copied_layout is not fig1.layout

    # Original layout should still have fig1 as parent
    assert fig1.layout.parent is fig1

    # Copied layout should have no parent
    assert copied_layout.parent is None