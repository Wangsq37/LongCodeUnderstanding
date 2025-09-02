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

import pytest
import random
from stellargraph.core.experimental import experimental, ExperimentalWarning

# some random data to check args are being passed through correctly
@pytest.fixture
def args():
    # Augmented: Edge cases, large, negative, float, string, zero.
    return (
        0,
        -123456789,
        3.1415926535,
        "",
        "test_string",
        999999999999,
    )


@pytest.fixture
def kwargs():
    # Augmented: various edge cases in the dict, empty string as key, None value
    return {
        "": None,
        "pi": 3.1415926535,
        "neg": -99,
        "long": "a" * 1024,
        "zero": 0,
        "list_val": [0, 1, 2],
    }


@experimental(reason="function is experimental", issues=[123, 456])
def func(*args, **kwargs):
    return args, kwargs


def test_experimental_func(args, kwargs):
    with pytest.warns(
        ExperimentalWarning,
        match=r"^func is experimental: function is experimental \(see: .*/123, .*/456\)\.",
    ):
        ret = func(*args, **kwargs)

    assert ret == (args, kwargs)


@experimental(reason="class is experimental", issues=[])
class ClassNoInit:
    pass


@experimental(reason="class is experimental")
class ClassInit:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_experimental_class(args, kwargs):
    with pytest.warns(
        ExperimentalWarning,
        match=r"^ClassNoInit is experimental: class is experimental\.",
    ):
        ClassNoInit()

    with pytest.warns(
        ExperimentalWarning,
        match=r"^ClassInit is experimental: class is experimental\.",
    ):
        instance = ClassInit(*args, **kwargs)

    # Augmented: check for new edge-case inputs in .args/.kwargs
    assert instance.args == args
    assert instance.kwargs == kwargs


class Class:
    @experimental(reason="method is experimental")
    def method(self, *args, **kwargs):
        return self, args, kwargs


def test_experimental_method(args, kwargs):
    instance = Class()
    with pytest.warns(
        ExperimentalWarning,
        match=r"^Class\.method is experimental: method is experimental\.",
    ):
        ret = instance.method(*args, **kwargs)

    # Augmented: check that result contains new args, kwargs
    assert ret[0] is instance
    assert ret[1:] == (args, kwargs)