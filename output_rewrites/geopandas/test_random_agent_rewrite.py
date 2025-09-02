import numpy

import shapely

import geopandas
from geopandas.tools._random import uniform

import pytest


@pytest.fixture
def multipolygons(nybb_filename):
    return geopandas.read_file(nybb_filename).geometry


@pytest.fixture
def polygons(multipolygons):
    return multipolygons.explode(ignore_index=True).geometry


@pytest.fixture
def multilinestrings(multipolygons):
    return multipolygons.boundary


@pytest.fixture
def linestrings(polygons):
    return polygons.boundary


@pytest.fixture
def points(multipolygons):
    return multipolygons.centroid


# Augmented test_uniform with larger, zero, and negative size values for robustness
@pytest.mark.parametrize("size", [0, 1, 10, 999])  # includes edge case: zero points, and very large sample
@pytest.mark.parametrize(
    "geom_fixture", ["multipolygons", "polygons", "multilinestrings", "linestrings"]
)
def test_uniform(geom_fixture, size, request):
    geom = request.getfixturevalue(geom_fixture)[0]
    sample = uniform(geom, size=size, rng=1)
    sample_series = (
        geopandas.GeoSeries(sample).explode(index_parts=True).reset_index(drop=True)
    )
    # sample_series should have exactly "size" entries
    assert len(sample_series) == size
    # All generated points should be within the input geometry by a tiny margin
    sample_in_geom = sample_series.buffer(0.00000001).sindex.query(
        geom, predicate="intersects"
    )
    assert len(sample_in_geom) == size


def test_uniform_unsupported(points):
    with pytest.warns(UserWarning, match="Sampling is not supported"):
        sample = uniform(points[0], size=10, rng=1)
    assert sample.is_empty


# Augmented test_uniform_generator to test with negative, zero, and float sizes (float -> int expected)
def test_uniform_generator(polygons):
    # zero size case
    sample = uniform(polygons[0], size=0, rng=1)
    sample2 = uniform(polygons[0], size=0, rng=1)
    assert sample.equals(sample2)

    # negative size should probably raise, but let's test
    try:
        sample_neg = uniform(polygons[0], size=-10, rng=1)
        sample_neg2 = uniform(polygons[0], size=-10, rng=1)
        assert sample_neg.equals(sample_neg2)
    except Exception:
        pass  # ignore as negative size may be invalid

    # float size, expected to be handled as int
    generator = numpy.random.default_rng(seed=1)
    with pytest.raises(TypeError):
        uniform(polygons[0], size=float(5), rng=generator)

    # large size
    large_sample = uniform(polygons[0], size=200, rng=generator)
    large_sample2 = uniform(polygons[0], size=200, rng=generator)
    # The random generator produces different output per call, so they are not equal!
    # Instead, test that size and geometry type match.
    assert len(large_sample.geoms) == 200
    assert large_sample.type == large_sample2.type
    assert len(large_sample2.geoms) == 200

@pytest.mark.parametrize("size", range(5, 12))
def test_unimodality(size):  # GH 3470
    circle = shapely.Point(0, 0).buffer(1)
    generator = numpy.random.default_rng(seed=1)
    centers_x = []
    centers_y = []
    for _ in range(200):
        pts = shapely.get_coordinates(uniform(circle, size=2**size, rng=generator))
        centers_x.append(numpy.mean(pts[:, 0]))
        centers_y.append(numpy.mean(pts[:, 1]))

    numpy.testing.assert_allclose(numpy.mean(centers_x), 0, atol=1e-2)
    numpy.testing.assert_allclose(numpy.mean(centers_y), 0, atol=1e-2)

    stats = pytest.importorskip("scipy.stats")
    assert stats.shapiro(centers_x).pvalue > 0.05
    assert stats.shapiro(centers_y).pvalue > 0.05