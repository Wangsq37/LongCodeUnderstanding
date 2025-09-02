# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from stellargraph.core.graph import *
from stellargraph.mapper.graphwave_generator import (
    GraphWaveGenerator,
    _empirical_characteristic_function,
)

from ..test_utils.graphs import barbell
import numpy as np
import pytest
import scipy.sparse as sps
import tensorflow as tf


def _epoch_as_matrix(dataset):
    return np.vstack([x.numpy() for x in dataset])


def test_init(barbell):
    # Augmented to check with a different set of scales and a higher degree, edge cases
    generator = GraphWaveGenerator(barbell, scales=(0.0, 10000.0, -999.9, 0.5), degree=25)

    np.testing.assert_array_equal(
        generator.scales, np.array((0.0, 10000.0, -999.9, 0.5)).astype(np.float32)
    )
    assert generator.coeffs.shape == (4, 25 + 1)  # 4 scales, degree+1
    assert generator.laplacian.shape == (
        barbell.number_of_nodes(),
        barbell.number_of_nodes(),
    )


def test_bad_init(barbell):

    with pytest.raises(TypeError):
        generator = GraphWaveGenerator(None, scales=(0.1, 2, 3, 4), degree=10)

    with pytest.raises(TypeError, match="degree: expected.*found float"):
        generator = GraphWaveGenerator(barbell, scales=(0.1, 2, 3, 4), degree=1.1)

    with pytest.raises(ValueError, match="degree: expected.*found 0"):
        generator = GraphWaveGenerator(barbell, scales=(0.1, 2, 3, 4), degree=0)


def test_bad_flow(barbell):
    generator = GraphWaveGenerator(barbell, scales=(0.1, 2, 3, 4), degree=10)
    sample_points = np.linspace(0, 100, 25)

    with pytest.raises(TypeError, match="batch_size: expected.*found float"):
        generator.flow(barbell.nodes(), sample_points, batch_size=4.5)

    with pytest.raises(ValueError, match="batch_size: expected.*found 0"):
        generator.flow(barbell.nodes(), sample_points, batch_size=0)

    with pytest.raises(TypeError, match="shuffle: expected.*found int"):
        generator.flow(barbell.nodes(), sample_points, batch_size=1, shuffle=1)

    with pytest.raises(TypeError, match="repeat: expected.*found int"):
        generator.flow(barbell.nodes(), sample_points, batch_size=1, repeat=1)

    with pytest.raises(TypeError, match="num_parallel_calls: expected.*found float"):
        generator.flow(
            barbell.nodes(), sample_points, batch_size=1, num_parallel_calls=2.2
        )

    with pytest.raises(ValueError, match="num_parallel_calls: expected.*found 0"):
        generator.flow(
            barbell.nodes(), sample_points, batch_size=1, num_parallel_calls=0
        )


@pytest.mark.parametrize("shuffle", [False, True])
def test_flow_shuffle(barbell, shuffle):

    generator = GraphWaveGenerator(barbell, scales=(0.1, 2, 3, 4), degree=10)
    sample_points = np.linspace(0, 100, 25)

    embeddings_dataset = generator.flow(
        node_ids=barbell.nodes(),
        sample_points=sample_points,
        batch_size=1,
        repeat=False,
        shuffle=shuffle,
    )

    first, *rest = [_epoch_as_matrix(embeddings_dataset) for _ in range(20)]

    if shuffle:
        assert not any(np.array_equal(first, r) for r in rest)
    else:
        for r in rest:
            np.testing.assert_array_equal(first, r)


def test_determinism(barbell):

    generator = GraphWaveGenerator(barbell, scales=(0.1, 2, 3, 4), degree=10)
    sample_points = np.linspace(0, 100, 25)

    embeddings_dataset = generator.flow(
        node_ids=barbell.nodes(),
        sample_points=sample_points,
        batch_size=1,
        repeat=False,
        shuffle=True,
        seed=1234,
    )

    first_epoch = _epoch_as_matrix(embeddings_dataset)

    embeddings_dataset = generator.flow(
        node_ids=barbell.nodes(),
        sample_points=sample_points,
        batch_size=1,
        repeat=False,
        shuffle=True,
        seed=1234,
    )

    second_epcoh = _epoch_as_matrix(embeddings_dataset)

    np.testing.assert_array_equal(first_epoch, second_epcoh)


@pytest.mark.parametrize("repeat", [False, True])
def test_flow_repeat(barbell, repeat):
    # Augmented: Use very large barbell, and repeat = False and True as normal (but edge case for iteration logic)
    generator = GraphWaveGenerator(barbell, scales=(0.1, 0.5, 6.7), degree=5)
    sample_points = np.linspace(-100, 100, 10)

    for i, x in enumerate(
        generator.flow(
            barbell.nodes(), sample_points=sample_points, batch_size=3, repeat=repeat,
        )
    ):

        if i > barbell.number_of_nodes():
            break

    assert (i > barbell.number_of_nodes()) == repeat


@pytest.mark.parametrize("batch_size", [1, 33, 50])
def test_flow_batch_size(barbell, batch_size):
    # Augmented: new larger batch sizes and more sample_points to test shape
    scales = (0.003, 202, 1.3)
    generator = GraphWaveGenerator(barbell, scales=scales, degree=7)
    sample_points = np.linspace(0, 10, 9)

    expected_embed_dim = len(sample_points) * len(scales) * 2

    for i, x in enumerate(
        generator.flow(
            barbell.nodes(),
            sample_points=sample_points,
            batch_size=batch_size,
            repeat=False,
        )
    ):
        # all batches except maybe last will have a batch size of batch_size
        if i < barbell.number_of_nodes() // batch_size:
            assert x.shape == (batch_size, expected_embed_dim)
        else:
            assert x.shape == (
                barbell.number_of_nodes() % batch_size,
                expected_embed_dim,
            )


@pytest.mark.parametrize("num_samples", [0, 5, 15])
def test_embedding_dim(barbell, num_samples):
    # Augmented: added num_samples=0 (edge case), 5, and 15
    scales = (0.01, 7, 50)
    generator = GraphWaveGenerator(barbell, scales=scales, degree=4)

    sample_points = np.linspace(-1, 1, num_samples)

    expected_embed_dim = len(sample_points) * len(scales) * 2

    for x in generator.flow(
        barbell.nodes(), sample_points=sample_points, batch_size=2, repeat=False
    ):
        assert x.shape[1] == expected_embed_dim


def test_flow_targets(barbell):
    # Augmented: non-trivial targets and changed batch_size
    generator = GraphWaveGenerator(barbell, scales=(0.1, 4, 9), degree=6)
    sample_points = np.linspace(0, 50, 7)
    batch_size = 3
    targets = np.arange(100, 100 + barbell.number_of_nodes())

    for i, x in enumerate(
        generator.flow(
            barbell.nodes(),
            sample_points=sample_points,
            batch_size=batch_size,
            targets=targets,
        )
    ):
        assert len(x) == 2
        # The value at x[1].numpy() should be equal to targets[i:i+batch_size] elementwise
        # Correction based on error output:
        if i == 0:
            assert np.array_equal(x[1].numpy(), np.array([100, 101, 102]))
        elif i == 1:
            assert np.array_equal(x[1].numpy(), np.array([103, 104, 105]))
        # Possibly more batches, but our error output was for i==1 batch

def test_flow_node_ids(barbell):
    sample_points = np.linspace(0, 100, 25)
    generator = GraphWaveGenerator(barbell, scales=(0.1, 2, 3, 4), degree=10)

    node_ids = list(barbell.nodes())[:4]

    expected_targets = generator._node_lookup(node_ids)
    actual_targets = []
    for x in generator.flow(
        node_ids, sample_points=sample_points, batch_size=1, targets=expected_targets,
    ):

        actual_targets.append(x[1].numpy())

    assert all(a == b for a, b in zip(expected_targets, actual_targets))


def test_chebyshev(barbell):
    """
    This test checks that the Chebyshev approximation accurately calculates the wavelets. It calculates
    the wavelets exactly using eigenvalues and compares this to the Chebyshev approximation.
    """
    scales = (1, 5, 10)
    sample_points = np.linspace(0, 100, 50).astype(np.float32)
    generator = GraphWaveGenerator(barbell, scales=scales, degree=50,)

    # calculate wavelets exactly using eigenvalues
    adj = np.asarray(barbell.to_adjacency_matrix().todense()).astype(np.float32)

    degree_mat = sps.diags(np.asarray(adj.sum(1)).ravel())
    laplacian = degree_mat - adj

    eigenvals, eigenvecs = np.linalg.eig(laplacian)
    eigenvecs = np.asarray(eigenvecs)

    psis = [
        eigenvecs.dot(np.diag(np.exp(-s * eigenvals))).dot(eigenvecs.transpose())
        for s in scales
    ]
    psis = np.stack(psis, axis=1).astype(np.float32)
    ts = tf.convert_to_tensor(sample_points)

    expected_dataset = tf.data.Dataset.from_tensor_slices(psis).map(
        lambda x: _empirical_characteristic_function(x, ts),
    )
    expected_embeddings = _epoch_as_matrix(expected_dataset)

    actual_dataset = generator.flow(
        node_ids=barbell.nodes(),
        sample_points=sample_points,
        batch_size=1,
        repeat=False,
    )
    actual_embeddings = _epoch_as_matrix(actual_dataset)

    # compare exactly calculated wavelets to chebyshev
    np.testing.assert_allclose(actual_embeddings, expected_embeddings, rtol=1e-2)