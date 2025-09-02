# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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

import numpy as np
import pytest

from stellargraph import IndexedArray


def test_indexed_array_empty():
    # Edge case: empty with a "shape" argument, and empty index explicitly as list, and shape with more than 2 axes
    frame = IndexedArray(np.empty((0, 3, 2)), index=[])
    assert frame.index == []
    np.testing.assert_array_equal(frame.values, np.empty((0, 3, 2)))


def test_indexed_array_non_empty():
    # Comprehensive: non-empty, floats, negative/zero indices, integer and string and range,
    # multi-dimensional, large input
    list_ids = ["", None, "long_string_123456"]
    array_ids = np.array([0.0, -99999.999, 999999.99])  # floats as "ids"
    range_ids = range(0, 30, 7)

    values = np.random.randint(-5, 20, size=(3, 2, 2))  # ints, negative values allowed

    # this test uses 'is' checks to validate that there's no copying of data
    frame = IndexedArray(values)
    assert frame.index == range(3)
    assert frame.values is values

    frame = IndexedArray(values, index=list_ids)
    assert frame.index is list_ids
    assert frame.values is values

    frame = IndexedArray(values, index=array_ids)
    assert frame.index is array_ids
    assert frame.values is values

    # Updated the input data for range_ids to match expected index length 3
    range_ids = range(0, 3, 1)
    frame = IndexedArray(values, index=range_ids)
    assert frame.index is range_ids
    assert frame.values is values


def test_indexed_array_invalid():
    values = np.random.rand(3, 4, 5)

    with pytest.raises(TypeError, match="values: expected a NumPy array .* found int"):
        IndexedArray(123)

    with pytest.raises(
        ValueError,
        match=r"values: expected an array with shape .* found shape \(\) of length 0",
    ):
        IndexedArray(np.zeros(()))

    with pytest.raises(
        ValueError,
        match=r"values: expected an array with shape .* found shape \(123,\) of length 1",
    ):
        IndexedArray(np.zeros(123))

    # check that the index `len`-failure works with or without index inference
    with pytest.raises(TypeError, match="index: expected a sequence .* found int"):
        IndexedArray(index=0)

    with pytest.raises(TypeError, match="index: expected a sequence .* found int"):
        IndexedArray(values, index=123)

    with pytest.raises(
        ValueError, match="values: expected the index length 2 .* found 3 rows"
    ):
        IndexedArray(values, index=range(0, 3, 2))