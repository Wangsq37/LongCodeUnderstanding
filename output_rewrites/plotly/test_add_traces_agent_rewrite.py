from unittest import TestCase

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from unittest.mock import MagicMock


class TestAddTracesMessage(TestCase):
    def setUp(self):
        # Construct initial scatter object
        self.figure = go.Figure(
            data=[
                go.Scatter(y=[3, 2, 1], marker={"color": "green"}),
                go.Bar(y=[3, 2, 1, 0, -1], marker={"opacity": 0.5}),
            ],
            layout={"xaxis": {"range": [-1, 4]}},
            frames=[go.Frame(layout={"yaxis": {"title": "f1"}})],
        )

        # Mock out the message method
        self.figure._send_addTraces_msg = MagicMock()

    def test_add_trace(self):
        # Add a trace
        self.figure.add_trace(go.Sankey(arrangement="snap"))

        # Check access properties
        self.assertEqual(self.figure.data[-1].type, "sankey")
        self.assertEqual(self.figure.data[-1].arrangement, "snap")

        # Check message
        self.figure._send_addTraces_msg.assert_called_once_with(
            [{"type": "sankey", "arrangement": "snap"}]
        )

    def test_add_traces(self):
        # Add two traces
        self.figure.add_traces(
            [
                go.Sankey(arrangement="snap"),
                go.Histogram2dContour(line={"color": "cyan"}),
            ]
        )

        # Check access properties
        self.assertEqual(self.figure.data[-2].type, "sankey")
        self.assertEqual(self.figure.data[-2].arrangement, "snap")

        self.assertEqual(self.figure.data[-1].type, "histogram2dcontour")
        self.assertEqual(self.figure.data[-1].line.color, "cyan")

        # Check message
        self.figure._send_addTraces_msg.assert_called_once_with(
            [
                {"type": "sankey", "arrangement": "snap"},
                {"type": "histogram2dcontour", "line": {"color": "cyan"}},
            ]
        )


def test_add_trace_exclude_empty_subplots():
    # More robust test data: Use floats, negative and zero, empty string in y, 3x3 subplot grid
    fig = make_subplots(3, 3)
    fig.add_trace(go.Scatter(x=[-1.5, 0.0, 2.7], y=[0.0, "", 5.5]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[10, 2.5, -7.7, 10000], y=[0, -5, 2, 3]), row=3, col=3)
    # Add traces with exclude_empty_subplots set to true
    fig.add_trace(
        go.Scatter(x=[4, 0, -100], y=[-1, 23.3, 1e9]),
        row="all",
        col="all",
        exclude_empty_subplots=True,
    )
    assert len(fig.data) == 4
    assert fig.data[2]["xaxis"] == "x" and fig.data[2]["yaxis"] == "y"
    assert fig.data[3]["xaxis"] == "x9" and fig.data[3]["yaxis"] == "y9"


def test_add_trace_no_exclude_empty_subplots():
    # More robust test data: Use floats and very large numbers, 3x2 subplot grid
    fig = make_subplots(3, 2)
    fig.add_trace(go.Scatter(x=[float("inf"), -1e10], y=[-0.1, 0.0]), row=2, col=1)
    fig.add_trace(go.Scatter(x=[2.2, 3.3, 4.4, 5.5], y=[2, 1, 0, -1]), row=3, col=2)
    # Add traces without exclude_empty_subplots, should add to all subplots
    fig.add_trace(go.Scatter(x=[8, 9, 10], y=[-8, -9, -10]), row="all", col="all")
    assert len(fig.data) == 8
    assert fig.data[2]["xaxis"] == "x" and fig.data[2]["yaxis"] == "y"
    assert fig.data[3]["xaxis"] == "x2" and fig.data[3]["yaxis"] == "y2"
    assert fig.data[4]["xaxis"] == "x3" and fig.data[4]["yaxis"] == "y3"
    assert fig.data[5]["xaxis"] == "x4" and fig.data[5]["yaxis"] == "y4"
    assert fig.data[6]["xaxis"] == "x5" and fig.data[6]["yaxis"] == "y5"
    assert fig.data[7]["xaxis"] == "x6" and fig.data[7]["yaxis"] == "y6"


def test_add_trace_exclude_totally_empty_subplots():
    # Edge case: Use non-square subplots, add three different layout shapes
    fig = make_subplots(2, 3)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=2)
    fig.add_trace(go.Scatter(x=["a", "b", "c"], y=[1.1, 2.2, 3.3]), row=2, col=3)
    fig.add_shape(dict(type="rect", x0=-10, x1=10, y0=-10, y1=10), row=1, col=2)
    fig.add_shape(dict(type="line", x0=0, x1=2, y0=2, y1=0), row=1, col=3)
    fig.add_shape(dict(type="circle", x0=1, x1=2, y0=1, y1=2), row=2, col=2)
    # Add traces with exclude_empty_subplots as a list with different values
    fig.add_trace(
        go.Scatter(x=[0, 0], y=[999, -999]),
        row="all",
        col="all",
        exclude_empty_subplots=["foo", "bar", "baz"],
    )
    assert len(fig.data) == 6
    assert fig.data[2]["xaxis"] == "x2" and fig.data[2]["yaxis"] == "y2"
    assert fig.data[3]["xaxis"] == "x3" and fig.data[3]["yaxis"] == "y3"  # <-- update: expected "x6" to "x3"


def test_add_trace_no_exclude_totally_empty_subplots():
    # Edge case: 1x4 subplot grid, negative and large numbers everywhere, add layout to 2nd subplot
    fig = make_subplots(1, 4)
    fig.add_trace(go.Scatter(x=[-100, 0], y=[1000, -1000]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[5000, 9000, -500], y=[0, 0, 0]), row=1, col=4)
    fig.add_shape(dict(type="rect", x0=-20, x1=200, y0=-5, y1=50), row=1, col=2)
    # Add traces without exclude_empty_subplots; should add to all subplots
    fig.add_trace(go.Scatter(x=[22, -33], y=[44, -55]), row="all", col="all")
    assert len(fig.data) == 6
    assert fig.data[2]["xaxis"] == "x" and fig.data[2]["yaxis"] == "y"
    assert fig.data[3]["xaxis"] == "x2" and fig.data[3]["yaxis"] == "y2"
    assert fig.data[4]["xaxis"] == "x3" and fig.data[4]["yaxis"] == "y3"
    assert fig.data[5]["xaxis"] == "x4" and fig.data[5]["yaxis"] == "y4"