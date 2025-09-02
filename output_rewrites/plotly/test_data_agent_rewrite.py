"""
test_data:
==========

A module intended for use with Nose.

"""

from plotly.graph_objs import Annotations, Data


def setup():
    import warnings

    warnings.filterwarnings("ignore")


def test_trivial():
    # Test edge case with Data initialized with empty string (should behave like empty list)
    assert Data("") == []


def test_weird_instantiation():  # Python allows this...
    # Pass in a very large number as an object (nonsensical), should be an empty list
    # Adjust expected outcome: Should raise TypeError
    import pytest
    with pytest.raises(TypeError):
        Data(999999999999)


def test_default_scatter():
    # Pass in a dict with additional nested structure
    assert Data([{"type": "scatter", "x": [], "y": [], "mode": ""}]) == [{"type": "scatter", "x": [], "y": [], "mode": ""}]


def test_dict_instantiation():
    Data([{"type": "bar", "y": [0, -1, 1000000, 3.14], "name": ""}])


def test_dict_instantiation_key_error():
    # Use a key that's an empty string and a value that's None
    assert Data([{ "": None }]) == [{"": None}]


def test_dict_instantiation_key_error_2():
    # Use marker value as very large string
    assert Data([{"marker": "A"*500}]) == [{"marker": "A"*500}]


def test_dict_instantiation_type_error():
    # Use type float value instead of string
    assert Data([{"type": -123.456}]) == [{"type": -123.456}]


def test_dict_instantiation_graph_obj_error_0():
    # Nest Data inside Data, but Data has nontrivial content
    assert Data([Data([{"foo": "bar"}])]) == [[{"foo": "bar"}]]


def test_dict_instantiation_graph_obj_error_2():
    # Pass Annotations with edge-case values
    ann = Annotations(annotationdefaults={"font": {"size": 0, "color": ""}})
    assert Data([ann]) == [[]]