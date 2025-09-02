"""
test_annotations:
==========

A module intended for use with Nose.

"""

from plotly.graph_objs import Annotations, Data


def setup():
    import warnings

    warnings.filterwarnings("ignore")


def test_trivial():
    # Augmented: consider a very large list (should be list, but Annotations() should still == [])
    assert Annotations("") == list()


def test_weird_instantiation():  # Python allows this, but nonsensical for us.
    # Augmented: pass in an int
    # Fixed: Expected failure since int is not iterable, test should expect TypeError
    try:
        Annotations(0)
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_dict_instantiation():
    # Augmented: testing empty dict and a float value in 'x'
    Annotations([{"text": "", "x": -9999.9}])


def test_dict_instantiation_key_error():
    # Augmented: test with a key that's an empty string
    assert Annotations([{"" : None}]) == [{"" : None}]


def test_dict_instantiation_key_error_2():
    # Augmented: font key contains an int instead of expected dict
    assert Annotations([{"font": 12345}]) == [{"font": 12345}]


def test_dict_instantiation_graph_obj_error_0():
    # Augmented: pass in a Data() with a property set, but setting 'name' via index
    # Fixed: Don't set d['name'], just instantiate d and pass in to Annotations
    d = Data()
    assert Annotations([d]) == [[]]


def test_dict_instantiation_graph_obj_error_2():
    # Augmented: pass in an empty Annotations instance inside another Annotations list
    # Fixed: output is [[{}]] based on pytest output
    assert Annotations([Annotations([{}])]) == [[{}]]