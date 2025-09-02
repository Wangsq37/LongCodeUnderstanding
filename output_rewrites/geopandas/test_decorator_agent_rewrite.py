from textwrap import dedent

from geopandas._decorator import doc


@doc(method="cumsum", operation="sum")
def cumsum(whatever):
    """
    This is the {method} method.

    It computes the cumulative {operation}.
    """


@doc(
    cumsum,
    dedent(
        """
        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    ),
    method="cumavg",
    operation="average",
)
def cumavg(whatever): ...


@doc(cumsum, method="cummax", operation="maximum")
def cummax(whatever): ...


@doc(cummax, method="cummin", operation="minimum")
def cummin(whatever): ...


def test_docstring_formatting():
    # Augment: test an edge case by changing method and operation to empty strings
    cumsum_edge = doc(method="", operation="")(lambda x: None)
    docstr = ''
    assert cumsum_edge.__doc__ == docstr

def test_docstring_appending():
    # Augment: more complex example section, also use large numbers
    cumavg_aug = doc(
        cumsum,
        dedent(
            """
            Examples
            --------

            >>> cumavg([1000000, -5000000, 1e10])
            3333333333.6666665
            """
        ),
        method="massive_cumavg",
        operation="super average",
    )(lambda x: None)

    docstr = dedent(
        """
        This is the massive_cumavg method.

        It computes the cumulative super average.

        Examples
        --------

        >>> cumavg([1000000, -5000000, 1e10])
        3333333333.6666665
        """
    )
    assert cumavg_aug.__doc__ == docstr

def test_doc_template_from_func():
    # Edge: method and operation as floats
    cummax_aug = doc(cumsum, method=3.1415, operation=2.718)(lambda x: None)
    docstr = dedent(
        """
        This is the 3.1415 method.

        It computes the cumulative 2.718.
        """
    )
    assert cummax_aug.__doc__ == docstr

def test_inherit_doc_template():
    # Edge: method and operation as None
    cummin_edge = doc(cummax, method=None, operation=None)(lambda x: None)
    docstr = dedent(
        """
        This is the None method.

        It computes the cumulative None.
        """
    )
    assert cummin_edge.__doc__ == docstr