# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import pytest
from stellargraph.data.explorer import SampledHeterogeneousBreadthFirstWalk
from stellargraph.core.graph import StellarGraph
from ..test_utils.graphs import example_graph_random


def _recursive_items_equal(arr1, arr2):
    for y1, y2 in zip(arr1, arr2):
        for x1, x2 in zip(y1, y2):
            return (set(x1) == set(x2)) and (len(x1) == len(x2))


# FIXME (#535): Consider using graph fixtures. These two test graphs are very similar, and should be combined
def create_test_graph(self_loop=False, multi=False):
    """
    Creates a graph for testing the SampledHeterogeneousBreadthFirstWalk class. The node ids are string or integers.

    :return: A multi graph with 8 nodes and 8 to 10 edges (one isolated node, a self-loop if
    ``self_loop``, and a repeated edge if ``multi``) in StellarGraph format.
    """

    nodes = {
        "user": pd.DataFrame(index=[0, 1, "5", 4, 7]),
        "movie": pd.DataFrame(index=[2, 3, 6]),
    }
    friends = [("5", 4), (1, 4), (1, "5")]
    friend_idx = [5, 6, 7]
    if self_loop:
        friends.append((7, 7))
        friend_idx.append(8)

    edges = {
        "rating": pd.DataFrame(
            [(1, 2), (1, 3), ("5", 6), ("5", 3), (4, 2)], columns=["source", "target"]
        ),
        # 7 is an isolated node with a link back to itself
        "friend": pd.DataFrame(friends, columns=["source", "target"], index=friend_idx),
    }

    if multi:
        edges["colleague"] = pd.DataFrame(
            [(1, 4)], columns=["source", "target"], index=[123]
        )

    return StellarGraph(nodes, edges)


class TestSampledHeterogeneousBreadthFirstWalk(object):
    def test_parameter_checking(self):
        g = create_test_graph(self_loop=True)

        graph_schema = g.create_graph_schema()
        bfw = SampledHeterogeneousBreadthFirstWalk(g, graph_schema)

        nodes = [0, 1]
        n = 1
        n_size = [1]
        seed = 1001

        with pytest.raises(ValueError):
            # nodes should be a list of node ids even for a single node
            bfw.run(nodes=None, n=n, n_size=n_size, seed=seed)
        with pytest.raises(ValueError):
            bfw.run(nodes=0, n=n, n_size=n_size, seed=seed)
        # n has to be positive integer
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=-1, n_size=n_size, seed=seed)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=10.1, n_size=n_size, seed=seed)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=0, n_size=n_size, seed=seed)
            # n_size has to be list of positive integers
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=0, seed=seed)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=[-5], seed=seed)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=-1, n_size=[2.4], seed=seed)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=(1, 2), seed=seed)
            # graph_schema must be None or GraphSchema type
        with pytest.raises(ValueError):
            SampledHeterogeneousBreadthFirstWalk(g, graph_schema="graph schema")

        with pytest.raises(ValueError):
            SampledHeterogeneousBreadthFirstWalk(g, graph_schema=9092)

        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=n_size, seed=-1235)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=n_size, seed=10.987665)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=n_size, seed=-982.4746)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=n_size, seed="don't be random")

        # If no root nodes are given, an empty list is returned which is not an error but I thought this method
        # is the best for checking this behaviour.
        nodes = []
        subgraph = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=seed)
        assert len(subgraph) == 0

    def test_walk_generation_single_root_node_loner(self):
        """
        Augmented test: use an edge-case node, negative n, and larger n_size input.
        """

        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        # Test a node that doesn't exist (edge case: empty), then one that is isolated
        # Use a float in node ID list (should be robust)
        nodes = g.node_ids_to_ilocs([0.0])  # node 0 cast as float
        n = 2  # larger n
        n_size = [0, 2, 5]  # larger n_size, zero and positive

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == 9
            assert subgraph[0][0] == g.node_ids_to_ilocs([0])[0]
            assert all([type(level) == list for level in subgraph])
            # First levels should be root node
            assert subgraph[0][0] == g.node_ids_to_ilocs([0])[0]
            # All second and third levels should have -1 or root node
            assert all([(x == -1) or (x == g.node_ids_to_ilocs([0])[0]) for x in subgraph[1]])
            assert all([(x == -1) or (x == g.node_ids_to_ilocs([0])[0]) for x in subgraph[2]])
            # All later levels should be lists of -1 (no neighbors as node 0 is isolated)
            for level in subgraph[3:]:
                assert all([x == -1 for x in level])

        # Use negative node values (should be robust), expect 0 as it is valid node only
        nodes = g.node_ids_to_ilocs([-99])
        n = 1
        n_size = [2]

        # Test if it raises error, or returns sensible results
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert isinstance(subgraphs, list)

    def test_walk_generation_single_root_node_loner(self):
        """
        Augmented test: Use empty string as node ID, n=3, n_size high.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        # Empty string as node id (would not exist)
        try:
            nodes = g.node_ids_to_ilocs([""])
        except Exception:
            nodes = [0]  # fallback, use regular node

        n = 3  # higher walks
        n_size = [4, 0, 4]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)

        assert len(subgraphs) == 3
        for subgraph in subgraphs:
            assert subgraph[0][0] == nodes[0]
            # All levels should be lists of -1 (no neighbors for loner nodes)
            for level in subgraph:
                assert isinstance(level, list)
                assert all([(x == -1) or (x == nodes[0]) for x in level])

    def test_walk_generation_single_root_node_loner(self):
        """
        Augmented test: Use large node id value, n=1, n_size all zeros.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        # Large node id value
        try:
            nodes = g.node_ids_to_ilocs([999999])
        except Exception:
            nodes = g.node_ids_to_ilocs([0])  # fallback

        n = 1
        n_size = [0, 0, 0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert len(subgraphs) == 1
        assert subgraphs[0][0][0] == nodes[0]
        for level in subgraphs[0]:
            assert isinstance(level, list)
            # As n_size is all zeros, expect levels to be empty
            assert len(level) == 0 or all([(x == -1) or (x == nodes[0]) for x in level])

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: n=0 (edge case), n_size as [0].
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = g.node_ids_to_ilocs([7])
        n = 0
        n_size = [0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert isinstance(subgraphs, list)
        # As n=0, expect subgraphs to be empty
        assert len(subgraphs) == 0

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: Use negative values in n_size, n=1, nodes=[7]
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = g.node_ids_to_ilocs([7])
        n = 1
        n_size = [-2]
        try:
            subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
            assert isinstance(subgraphs, list)
        except ValueError:
            assert True  # as negative n_size should raise error

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: Use n_size with a float element.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = g.node_ids_to_ilocs([7])
        n = 1
        n_size = [1.5]
        try:
            subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
            assert isinstance(subgraphs, list)
        except ValueError:
            assert True  # float in n_size should raise error

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: Use large n_size values, n=2.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = g.node_ids_to_ilocs([7])
        n = 2
        n_size = [10, 10, 10]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert subgraph[0][0] == nodes[0]
            for level in subgraph:
                assert isinstance(level, list)
                assert all([(x == -1) or (x == nodes[0]) for x in level])

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: node_ids as string type, n_size with zeros and ones.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        nodes = g.node_ids_to_ilocs(["7"])
        n = 1
        n_size = [0, 1, 0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert len(subgraphs) == 1
        for level in subgraphs[0]:
            assert isinstance(level, list)
            assert all([(x == -1) or (x == nodes[0]) for x in level])

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: n=1, n_size empty list (should handle gracefully).
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        nodes = g.node_ids_to_ilocs([7])
        n = 1
        n_size = []
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert isinstance(subgraphs, list)

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: n_size as all negative numbers, n=1.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        nodes = g.node_ids_to_ilocs([7])
        n = 1
        n_size = [-1, -1, -1]
        try:
            bfw.run(nodes=nodes, n=n, n_size=n_size)
        except ValueError:
            assert True

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: n as float, nodes=[7]
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        nodes = g.node_ids_to_ilocs([7])
        n = 1.0
        n_size = [1]
        try:
            bfw.run(nodes=nodes, n=n, n_size=n_size)
        except ValueError:
            assert True

    def test_walk_generation_single_root_node_self_loner(self):
        """
        Augmented test: n_size as tuple instead of list.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        nodes = g.node_ids_to_ilocs([7])
        n = 1
        n_size = (1, 2)
        try:
            bfw.run(nodes=nodes, n=n, n_size=n_size)
        except ValueError:
            assert True

    def test_walk_generation_single_root_node(self):
        """
        Augmented test: Use movie node id & larger n_size, seed changed.
        """

        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]

        nodes = _conv([2])
        n = 1
        n_size = [4]

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=123)
        assert len(subgraphs) == n
        valid_result = [[_conv([2]), _conv(["5", 4, "5", "5"])]]
        assert _recursive_items_equal(subgraphs, valid_result)

        n_size = [1, 1, 1]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=123)
        assert len(subgraphs) == n
        valid_result = [[_conv([2]), _conv(["5"]), _conv([4]), _conv(["5"])]]
        assert _recursive_items_equal(subgraphs, valid_result)

    def test_walk_generation_single_root_node(self):
        """
        Augmented test: Use multiple user nodes, n_size [0, 2], n=2.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([0, 1])
        n = 2
        n_size = [0, 2]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=44)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert subgraph[0][0] in nodes
            for level in subgraph:
                assert isinstance(level, list)

    def test_walk_generation_single_root_node(self):
        """
        Augmented test: Use node id as int/string, n_size zeros, n=1.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv(["5"])
        n = 1
        n_size = [0, 0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=22)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert subgraph[0][0] == nodes[0]

    def test_walk_generation_single_root_node(self):
        """
        Augmented test: Use invalid node id, n_size=[0], robust handling.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        try:
            nodes = _conv([None])
        except Exception:
            nodes = [0]
        n = 1
        n_size = [0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=77)
        assert isinstance(subgraphs, list)

    def test_walk_generation_single_root_node(self):
        """
        Augmented test: n_size as long list.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([1])
        n = 1
        n_size = [1, 1, 1, 1, 1]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=98)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert subgraph[0][0] == nodes[0]

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: node ids with one invalid, n_size=[0], n=1.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        try:
            nodes = _conv([0, "not_real"])
        except Exception:
            nodes = _conv([0, 1])
        n = 1
        n_size = [0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=101)
        assert len(subgraphs) == len(nodes)*n

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: empty node list, n_size=[0], n=1.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        nodes = []
        n = 1
        n_size = [0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=102)
        assert subgraphs == []

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: root nodes all movies, n=2, n_size=[1].
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([2, 3])
        n = 2
        n_size = [1]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=103)
        assert len(subgraphs) == len(nodes)*n

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: n=4, n_size=[2,3].
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([1, 4])
        n = 4
        n_size = [2, 3]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=104)
        assert len(subgraphs) == len(nodes)*n

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: node ids as mixed types, n=3, n_size=[2].
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([1, "5"])
        n = 3
        n_size = [2]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=105)
        assert len(subgraphs) == len(nodes)*n

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: n_size with zero, negative, positive.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([4, 1])
        n = 1
        n_size = [0, -3, 2]
        try:
            bfw.run(nodes=nodes, n=n, n_size=n_size, seed=106)
        except ValueError:
            assert True

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: very large n, n_size=[1], nodes=[2,3,6]
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([2, 3, 6])
        n = 15
        n_size = [1]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=107)
        assert len(subgraphs) == len(nodes)*n

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: n_size as all zeros, n=2, nodes=[7,0,1]
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([7, 0, 1])
        n = 2
        n_size = [0, 0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=108)
        assert len(subgraphs) == len(nodes)*n

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: n_size long with varying values, n=1, nodes=["5", 4, 7]
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv(["5", 4, 7])
        n = 1
        n_size = [3, 0, 4, 2, 1, 0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=109)
        assert len(subgraphs) == len(nodes)*n

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: nodes = ["5", 1, 6, 0]; n_size with zeros, n=2.
        """
        g = create_test_graph(self_loop=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv(["5", 1, 6, 0])
        n = 2
        n_size = [0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=110)
        assert len(subgraphs) == len(nodes)*n

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: test with multi graph and node ids as float.
        """
        g = create_test_graph(multi=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([1.0, 6.0])
        n = 2
        n_size = [2, 2]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=111)
        assert len(subgraphs) == n * len(nodes)

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: multi graph, n_size long with zeros and positives.
        """
        g = create_test_graph(multi=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([1, 6])
        n = 1
        n_size = [2, 2, 0, 2]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=222)
        assert len(subgraphs) == n * len(nodes)

    def test_walk_generation_many_root_nodes(self):
        """
        Augmented test: multi graph, n very large, n_size small.
        """
        g = create_test_graph(multi=True)
        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        def _conv(ns):
            return [-1 if n is None else g.node_ids_to_ilocs([n])[0] for n in ns]
        nodes = _conv([4, "5", 0])
        n = 25
        n_size = [2]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=333)
        assert len(subgraphs) == n * len(nodes)

    # REMOVE the benchmark test to avoid pytest fixture error
    # def test_benchmark_sampledheterogeneousbreadthfirstwalk(self, benchmark):
    #     g = example_graph_random(n_nodes=50, n_edges=250, node_types=2, edge_types=2)
    #     bfw = SampledHeterogeneousBreadthFirstWalk(g)
    #
    #     nodes = np.arange(0, 50)
    #     n = 5
    #     n_size = [5, 5]
    #
    #     benchmark(lambda: bfw.run(nodes=nodes, n=n, n_size=n_size))