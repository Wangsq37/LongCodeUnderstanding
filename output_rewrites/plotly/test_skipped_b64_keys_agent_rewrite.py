import json
from unittest import TestCase
import numpy as np
from ...test_optional.optional_utils import NumpyTestUtilsMixin
import plotly.graph_objs as go


class TestShouldNotUseBase64InUnsupportedKeys(NumpyTestUtilsMixin, TestCase):
    def test_np_geojson(self):
        # Edge case: empty coordinates, negative coordinates, larger array
        normal_coordinates = [
            [
                [0, 0],
                [-120.5, 85.354],
                [1e6, -1e6],
                [42, -42],
                []
            ]
        ]

        numpy_coordinates = np.array(normal_coordinates, dtype=object)

        data = [
            {
                "type": "choropleth",
                "locations": ["XX"],
                "featureidkey": "properties.id",
                "z": np.array([-5.5, 0, 999999999]),
                "geojson": {
                    "type": "Feature",
                    "properties": {"id": "XX"},
                    "geometry": {"type": "Polygon", "coordinates": numpy_coordinates},
                },
            }
        ]

        fig = go.Figure(data=data)

        assert (
            json.loads(fig.to_json())["data"][0]["geojson"]["geometry"]["coordinates"]
            == normal_coordinates
        )

    def test_np_layers(self):
        # Edge case: negative and float dash values, extreme coordinates
        layout = {
            "mapbox": {
                "layers": [
                    {
                        "sourcetype": "geojson",
                        "type": "line",
                        "line": {"dash": np.array([-3.75, 0.0, 1000000])},
                        "source": {
                            "type": "FeatureCollection",
                            "features": [
                                {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "LineString",
                                        "coordinates": np.array(
                                            [[-180, 90.0], [180, -90.0], [0, 0]]
                                        ),
                                    },
                                }
                            ],
                        },
                    },
                ],
                "center": {"lon": 180, "lat": -90},
            },
        }
        data = [{"type": "scattermap"}]

        fig = go.Figure(data=data, layout=layout)

        assert (fig.layout["mapbox"]["layers"][0]["line"]["dash"] == (-3.75, 0.0, 1000000)).all()

        assert json.loads(fig.to_json())["layout"]["mapbox"]["layers"][0]["source"][
            "features"
        ][0]["geometry"]["coordinates"] == [[-180, 90.0], [180, -90.0], [0, 0]]

    def test_np_range(self):
        # Edge case: large range, including negative, float and integer
        layout = {"xaxis": {"range": np.array([-1000.5, 1000])}}

        fig = go.Figure(data=[{"type": "scatter"}], layout=layout)

        assert json.loads(fig.to_json())["layout"]["xaxis"]["range"] == [-1000.5, 1000]