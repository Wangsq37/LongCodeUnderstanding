"""
Test that our implementation of wrap_non_picklable_objects mimics
properly the loky implementation.
"""

from .._cloudpickle_wrapper import (
    _my_wrap_non_picklable_objects,
    wrap_non_picklable_objects,
)


def a_function(x):
    return x


class AClass(object):
    def __call__(self, x):
        return x


def test_wrap_non_picklable_objects():
    # Enhanced test: test a variety of input types and edge cases,
    # including empty string, zero, negative, float, large int,
    # tuple, list, dict, None.
    test_values = [
        0,
        -17,
        3.1415,
        12345678901234567890,  # very large int
        "",
        "string value",
        [],
        [1, 2, 3],
        (),
        (42,),
        {},
        {"key": "value", "num": 99},
        None
    ]
    for obj in (a_function, AClass()):
        for value in test_values:
            wrapped_obj = wrap_non_picklable_objects(obj)
            my_wrapped_obj = _my_wrap_non_picklable_objects(obj)
            assert wrapped_obj(value) == my_wrapped_obj(value)