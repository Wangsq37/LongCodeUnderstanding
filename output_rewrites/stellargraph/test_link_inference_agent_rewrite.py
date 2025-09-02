# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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

"""
Tests for link inference functions
"""

from stellargraph.layer.link_inference import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pytest


def make_orthonormal_vectors(dim):
    x_src = np.random.randn(dim)
    x_src /= np.linalg.norm(x_src)  # normalize x_src
    x_dst = np.random.randn(dim)
    x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
    x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

    # Check the IP is zero for numpy operations
    assert np.dot(x_src, x_dst) == pytest.approx(0)

    return x_src, x_dst


class Test_LinkEmbedding(object):
    """
    Group of tests for link_inference() function
    """

    d = 100  # dimensionality of embedding vector space
    d_out = 10  # dimensionality of link inference output

    def test_ip(self):
        """ Test the 'ip' binary operator on edge cases: zeros, negatives, large values """

        # 1. Zero vectors
        x_src = np.zeros(self.d)
        x_dst = np.zeros(self.d)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = LinkEmbedding(method="ip", activation="linear")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        # 2. Negative values
        x_src = -np.ones(self.d)
        x_dst = np.ones(self.d)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = LinkEmbedding(method="ip", activation="linear")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(-self.d, abs=1.5e-7)

        # 3. Large positive values
        x_src = np.full(self.d, 1e6)
        x_dst = np.full(self.d, 1e6)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = LinkEmbedding(method="ip", activation="linear")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(1e6 * 1e6 * self.d, abs=1e4)

        # Test sigmoid activation with zeros (should yield 0.5)
        x_src = np.zeros(self.d)
        x_dst = np.zeros(self.d)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = LinkEmbedding(method="ip", activation="sigmoid")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(0.5, abs=1.5e-7)

        # Test sigmoid activation with large positive values (should saturate near 1)
        x_src = np.full(self.d, 1e6)
        x_dst = np.full(self.d, 1e6)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = LinkEmbedding(method="ip", activation="sigmoid")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(1, abs=1e-7)

    def test_ip_single_tensor(self):
        """ Test the 'ip' binary operator on edge cases using single tensor """

        # Test with zero and negative input values
        x_src = np.zeros(self.d)
        x_dst = -np.ones(self.d)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        x_link_sd = tf.stack([x_src_tf, x_dst_tf], axis=1)
        x_link_ss = tf.stack([x_src_tf, x_src_tf], axis=1)

        li = LinkEmbedding(method="ip", activation="linear")(x_link_sd)
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        li = LinkEmbedding(method="ip", activation="linear")(x_link_ss)
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        # Test sigmoid activation with zeros (should yield 0.5)
        li = LinkEmbedding(method="ip", activation="sigmoid")(x_link_sd)
        assert li.numpy() == pytest.approx(0.5, abs=1.5e-7)

        li = LinkEmbedding(method="ip", activation="sigmoid")(x_link_ss)
        assert li.numpy() == pytest.approx(0.5, abs=1.5e-7)

    def test_mul_l1_l2_avg(self):
        """ Test the binary operators: 'mul'/'Hadamard', 'l1', 'l2', 'avg' with diverse values """

        # Edge input: alternating positive and negative large numbers
        x_src = np.array([1e5 if i % 2 == 0 else -1e5 for i in range(self.d)], dtype=float).reshape(1, 1, self.d)
        x_dst = np.array([-1e5 if i % 2 == 0 else 1e5 for i in range(self.d)], dtype=float).reshape(1, 1, self.d)

        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg"]:
            out = LinkEmbedding(method=op)([inp_src, inp_dst])
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)
            res = li.predict(x=[x_src, x_dst])
            assert res.shape == (1, 1, self.d)
            assert isinstance(res.flatten()[0], np.float32)

        for op in ["concat"]:
            out = LinkEmbedding(method=op)([inp_src, inp_dst])
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)
            res = li.predict(x=[x_src, x_dst])
            assert res.shape == (1, 1, 2 * self.d)
            assert isinstance(res.flatten()[0], np.float32)

    def test_mul_l1_l2_avg_single_tensor(self):
        """ Test the binary operators: 'mul'/'Hadamard', 'l1', 'l2', 'avg' with zeros and negative vectors """

        x_src = np.zeros(self.d).reshape(1, self.d)
        x_dst = -np.ones(self.d).reshape(1, self.d)
        x_link_np = np.stack([x_src, x_dst], axis=1)
        x_link = keras.Input(shape=(2, self.d))

        for op in ["mul", "l1", "l2", "avg"]:
            out = LinkEmbedding(method=op)(x_link)
            li = keras.Model(inputs=x_link, outputs=out)

            res = li.predict(x=x_link_np)
            assert res.shape == (1, self.d)
            assert isinstance(res.flatten()[0], np.float32)

        for op in ["concat"]:
            out = LinkEmbedding(method=op)(x_link)
            li = keras.Model(inputs=x_link, outputs=out)
            res = li.predict(x=x_link_np)
            assert res.shape == (1, 2 * self.d)
            assert isinstance(res.flatten()[0], np.float32)


class Test_Link_Inference(object):
    """
    Group of tests for link_inference() function
    """

    d = 100  # dimensionality of embedding vector space
    d_out = 10  # dimensionality of link inference output

    def test_ip(self):
        """ Test the 'ip' binary operator on edge cases """

        # Large vectors (ones and minus ones)
        x_src = np.ones(self.d)
        x_dst = -np.ones(self.d)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = link_inference(edge_embedding_method="ip", output_act="linear")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(-self.d, abs=1.5e-7)

        li = link_inference(edge_embedding_method="ip", output_act="linear")([x_src_tf, x_src_tf])
        assert li.numpy() == pytest.approx(self.d, abs=1.5e-7)

        # Test sigmoid activation, ones and minus ones
        li = link_classification(edge_embedding_method="ip", output_act="sigmoid")([x_src_tf, x_dst_tf])
        # sigmoid(-self.d), for large negative goes to 0
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        li = link_classification(edge_embedding_method="ip", output_act="sigmoid")([x_src_tf, x_src_tf])
        # sigmoid(self.d), for large positive goes to 1
        assert li.numpy() == pytest.approx(1, abs=1.5e-7)

    def test_mul_l1_l2_avg(self):
        """ Test the binary operators: 'mul'/'Hadamard', 'l1', 'l2', 'avg', 'concat' with various values """

        # Various values: float range
        x_src = np.linspace(-1e3, 1e3, self.d).reshape(1, 1, self.d)
        x_dst = np.linspace(1e3, -1e3, self.d).reshape(1, 1, self.d)

        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg", "concat"]:
            out = link_inference(output_dim=self.d_out, edge_embedding_method=op)([inp_src, inp_dst])
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)
            res = li.predict(x=[x_src, x_dst])
            assert res.shape == (1, self.d_out)
            assert isinstance(res.flatten()[0], np.float32)


class Test_Link_Classification(object):
    """
    Group of tests for link_classification() function
    """

    d = 100  # dimensionality of embedding vector space
    d_out = 10  # dimensionality of link classification output

    def test_ip(self):
        """ Test the 'ip' binary operator on edge cases """

        # Edge case: input vectors as zeros, then large positive
        x_src = np.zeros(self.d)
        x_dst = np.zeros(self.d)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = link_classification(edge_embedding_method="ip", output_act="linear")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        x_src = np.full(self.d, 1e6)
        x_dst = np.full(self.d, 1e6)
        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = link_classification(edge_embedding_method="ip", output_act="linear")([x_src_tf, x_src_tf])
        assert li.numpy()[0, 0] == pytest.approx(1e6 * 1e6 * self.d, abs=1e4)

        li = link_classification(edge_embedding_method="ip", output_act="sigmoid")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(1, abs=1e-7)

        li = link_classification(edge_embedding_method="ip", output_act="sigmoid")([x_src_tf, x_src_tf])
        assert li.numpy() == pytest.approx(1, abs=1e-7)

    def test_mul_l1_l2_avg(self):
        """ Test the binary operators: 'mul'/'Hadamard', 'l1', 'l2', 'avg', 'concat' with negative numbers """

        x_src = -np.ones(self.d).reshape(1, 1, self.d)
        x_dst = -np.ones(self.d).reshape(1, 1, self.d)
        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg", "concat"]:
            out = link_classification(output_dim=self.d_out, edge_embedding_method=op)([inp_src, inp_dst])
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)

            res = li.predict(x=[x_src, x_dst])
            assert res.shape == (1, self.d_out)
            assert isinstance(res.flatten()[0], np.float32)
            assert all(res.flatten() >= 0)
            assert all(res.flatten() <= 1)


class Test_Link_Regression(object):
    """
    Group of tests for link_regression() function
    """

    d = 100  # dimensionality of embedding vector space
    d_out = 10  # dimensionality of link classification output
    clip_limits = (0, 1)

    def test_ip(self):
        """ Test the 'ip' binary operator on edge cases """

        x_src = np.zeros(self.d)
        x_dst = np.ones(self.d)
        expected = np.dot(x_src, x_dst)

        x_src_tf = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst_tf = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = link_regression(edge_embedding_method="ip")([x_src_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        li = link_regression(edge_embedding_method="ip")([x_dst_tf, x_dst_tf])
        assert li.numpy() == pytest.approx(self.d, abs=1.5e-7)

    def test_mul_l1_l2_avg(self):
        """ Test the binary operators: 'mul'/'Hadamard', 'l1', 'l2', 'avg', 'concat' using negative and positive values """

        x_src = np.full(self.d, -2).reshape(1, 1, self.d)
        x_dst = np.full(self.d, 2).reshape(1, 1, self.d)

        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg", "concat"]:
            out = link_regression(output_dim=self.d_out, edge_embedding_method=op)([inp_src, inp_dst])
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)

            res = li.predict(x=[x_src, x_dst])
            assert res.shape == (1, self.d_out)
            assert isinstance(res.flatten()[0], np.float32)

    def test_clip_limits(self):
        """
        Test calling with the leaky clip thresholds with all positive values
        """

        print("\n Testing clip limits...")
        x_src = np.full(self.d, 5.0).reshape(1, 1, self.d)
        x_dst = np.full(self.d, 5.0).reshape(1, 1, self.d)

        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg", "concat"]:
            out = link_regression(
                output_dim=self.d_out,
                edge_embedding_method=op,
                clip_limits=self.clip_limits,
            )([inp_src, inp_dst])
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)

            res = li.predict(x=[x_src, x_dst])
            assert res.shape == (1, self.d_out)
            assert isinstance(res.flatten()[0], np.float32)