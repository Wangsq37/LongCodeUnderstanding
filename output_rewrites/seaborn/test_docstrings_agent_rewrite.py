from seaborn._docstrings import DocstringComponents


EXAMPLE_DICT = dict(
    param_large_number="""
num : int
    A very large number parameter.
    """,
    param_empty_string="""
empty : str
    An empty string parameter.
    """,
    param_negative_float="""
nfloat : float
    A negative floating point parameter.
    """,
)


class ExampleClass:
    def example_method(self):
        """An example method.

        Parameters
        ----------
        num : int
            A method parameter taking a large number.

        empty : str
            A method parameter that may be an empty string.

        nfloat : float
            A method parameter that may be negative.

        """


def example_func():
    """An example function.

    Parameters
    ----------
    num : int
        A function parameter with a large value.

    empty : str
        A function parameter that may be empty.

    nfloat : float
        A negative floating point.

    """


class TestDocstringComponents:

    def test_from_dict(self):
        # Test with edge cases: large number, empty string, negative float
        obj = DocstringComponents(EXAMPLE_DICT)
        assert obj.param_large_number == "num : int\n    A very large number parameter."
        assert obj.param_empty_string == "empty : str\n    An empty string parameter."
        assert obj.param_negative_float == "nfloat : float\n    A negative floating point parameter."

    def test_from_nested_components(self):
        obj_inner = DocstringComponents(EXAMPLE_DICT)
        obj_outer = DocstringComponents.from_nested_components(inner=obj_inner)
        assert obj_outer.inner.param_large_number == "num : int\n    A very large number parameter."
        assert obj_outer.inner.param_empty_string == "empty : str\n    An empty string parameter."
        assert obj_outer.inner.param_negative_float == "nfloat : float\n    A negative floating point parameter."

    def test_from_function(self):
        obj = DocstringComponents.from_function_params(example_func)
        assert obj.num == "num : int\n    A function parameter with a large value."
        assert obj.empty == "empty : str\n    A function parameter that may be empty."
        assert obj.nfloat == "nfloat : float\n    A negative floating point."

    def test_from_method(self):
        obj = DocstringComponents.from_function_params(
            ExampleClass.example_method
        )
        assert obj.num == "num : int\n    A method parameter taking a large number."
        assert obj.empty == "empty : str\n    A method parameter that may be an empty string."
        assert obj.nfloat == "nfloat : float\n    A method parameter that may be negative."