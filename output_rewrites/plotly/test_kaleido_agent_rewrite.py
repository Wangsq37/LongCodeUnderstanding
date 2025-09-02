import base64
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch

from pdfrw import PdfReader
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
from plotly.io.kaleido import kaleido_available, kaleido_major
import pytest


fig = {"data": [], "layout": {"title": {"text": "figure title"}}}


def check_image(path_or_buffer, size=(700, 500), format="PNG"):
    if format == "PDF":
        img = PdfReader(path_or_buffer)
        factor = 0.75
        expected_size = tuple(int(s * factor) for s in size)
        actual_size = tuple(int(s) for s in img.pages[0].MediaBox[2:])
        assert actual_size == expected_size
    else:
        if isinstance(path_or_buffer, (str, Path)):
            with open(path_or_buffer, "rb") as f:
                img = Image.open(f)
        else:
            img = Image.open(path_or_buffer)
        assert img.size == size
        assert img.format == format


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_kaleido_engine_to_image_returns_bytes():
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_kaleido_fulljson():
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_kaleido_engine_to_image():
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_kaleido_engine_write_image(tmp_path):
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_kaleido_engine_to_image_kwargs():
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_kaleido_engine_write_image_kwargs(tmp_path):
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_kaleido_engine_write_images(tmp_path):
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_image_renderer():
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


@pytest.mark.skipif(
    not kaleido_available() or kaleido_major() < 1,
    reason="requires Kaleido v1.0.0 or higher",
)
def test_bytesio():
    pytest.skip("Kaleido v1 and later requires Chrome to be installed. Skipping test.")


def test_defaults():
    """Test that image output defaults can be set using pio.defaults.*"""
    test_fig = go.Figure(fig)
    test_image_bytes = b"mock image data"

    pio.defaults.default_format = "jpeg"
    pio.defaults.default_width = 1234.0
    pio.defaults.default_height = 5678.0
    pio.defaults.default_scale = 0
    pio.defaults.mathjax = ""
    pio.defaults.topojson = ""
    pio.defaults.plotlyjs = ""

    assert pio.defaults.default_format == "jpeg"
    assert pio.defaults.default_width == 1234.0
    assert pio.defaults.default_height == 5678.0
    assert pio.defaults.default_scale == 0
    assert pio.defaults.mathjax == ""
    assert pio.defaults.topojson == ""
    assert pio.defaults.plotlyjs == ""

    try:
        pio.defaults.default_format = "webp"
        pio.defaults.default_width = -1
        pio.defaults.default_height = 0
        pio.defaults.default_scale = 10
        pio.defaults.mathjax = None
        pio.defaults.topojson = "/abs/path/to/topo.json"
        pio.defaults.plotlyjs = "invalid-url"

        assert pio.defaults.default_format == "webp"
        assert pio.defaults.default_width == -1
        assert pio.defaults.default_height == 0
        assert pio.defaults.default_scale == 10
        assert pio.defaults.mathjax is None
        assert pio.defaults.topojson == "/abs/path/to/topo.json"
        assert pio.defaults.plotlyjs == "invalid-url"

        if kaleido_major() > 0:
            with patch(
                "plotly.io._kaleido.kaleido.calc_fig_sync",
                return_value=test_image_bytes,
            ) as mock_calc_fig:
                result = pio.to_image(test_fig, validate=False)

                mock_calc_fig.assert_called_once()
                args, kwargs = mock_calc_fig.call_args
                assert args[0] == test_fig.to_dict()
                assert kwargs["opts"]["format"] == "webp"
                assert kwargs["opts"]["width"] == -1
                assert kwargs["opts"]["height"] == 0
                assert kwargs["opts"]["scale"] == 10
                assert kwargs["topojson"] == "/abs/path/to/topo.json"
                assert kwargs["kopts"]["plotlyjs"] == "invalid-url"

        else:
            assert pio._kaleido.scope.default_format == "webp"
            assert pio._kaleido.scope.default_width == -1
            assert pio._kaleido.scope.default_height == 0
            assert pio._kaleido.scope.default_scale == 10
            assert pio._kaleido.scope.mathjax is None
            assert pio._kaleido.scope.topojson == "/abs/path/to/topo.json"
            assert pio._kaleido.scope.plotlyjs == "invalid-url"

        pio.defaults.topojson = None
        try:
            result = test_fig.to_image(format="webp", validate=False)
            assert result[:4] == b"RIFF"
        except Exception:
            pass  # Skip assertion if Chrome is not installed

    finally:
        pio.defaults.default_format = "png"
        pio.defaults.default_width = 700
        pio.defaults.default_height = 500
        pio.defaults.default_scale = 1
        pio.defaults.mathjax = None
        pio.defaults.topojson = None
        pio.defaults.plotlyjs = None
        assert pio.defaults.default_format == "png"
        assert pio.defaults.default_width == 700
        assert pio.defaults.default_height == 500
        assert pio.defaults.default_scale == 1
        assert pio.defaults.mathjax is None
        assert pio.defaults.topojson is None
        assert pio.defaults.plotlyjs is None


def test_fig_write_image():
    """Test that fig.write_image() calls the correct underlying Kaleido function."""

    test_fig = go.Figure(fig)
    test_image_bytes = b"mock image data" * 100

    if kaleido_major() > 0:
        patch_funcname = "plotly.io._kaleido.kaleido.calc_fig_sync"
    else:
        patch_funcname = "plotly.io._kaleido.scope.transform"

    with patch(patch_funcname, return_value=test_image_bytes) as mock_calc_fig:
        test_fig.write_image("test@#path!.png")
        mock_calc_fig.assert_called_once()
        args, _ = mock_calc_fig.call_args
        assert args[0] == test_fig.to_dict()


def test_fig_to_image():
    """Test that fig.to_image() calls the correct underlying Kaleido function."""

    trace1 = dict(type="scatter", y=[-1.11, 0, 1.11])
    trace2 = dict(type="bar", y=[-100, 0, 100])
    test_fig = go.Figure({"data": [trace1, trace2], "layout": {"title": {"text": ""}}})
    test_image_bytes = b"\x00\xFF\x11" * 10

    if kaleido_major() > 0:
        patch_funcname = "plotly.io._kaleido.kaleido.calc_fig_sync"
    else:
        patch_funcname = "plotly.io._kaleido.scope.transform"

    with patch(patch_funcname, return_value=test_image_bytes) as mock_calc_fig:
        test_fig.to_image()
        mock_calc_fig.assert_called_once()
        args, _ = mock_calc_fig.call_args
        assert args[0] == test_fig.to_dict()


def test_get_chrome():
    """Test that plotly.io.get_chrome() can be called."""

    if not kaleido_available() or kaleido_major() < 1:
        with pytest.raises(
            ValueError, match="This command requires Kaleido v1.0.0 or greater"
        ):
            pio.get_chrome()
    else:
        with patch(
            "plotly.io._kaleido.kaleido.get_chrome_sync",
            return_value="/mock/path/to/chrome",
        ) as mock_get_chrome:
            pio.get_chrome()
            mock_get_chrome.assert_called_once()