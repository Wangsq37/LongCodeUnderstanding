"""
Test the func_inspect module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import functools

from joblib.func_inspect import (
    _clean_win_chars,
    filter_args,
    format_signature,
    get_func_code,
    get_func_name,
)
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises


###############################################################################
# Module-level functions and fixture, for tests
def f(x, y=0):
    pass


def g(x):
    pass


def h(x, y=0, *args, **kwargs):
    pass


def i(x=1):
    pass


def j(x, y, **kwargs):
    pass


def k(*args, **kwargs):
    pass


def m1(x, *, y):
    pass


def m2(x, *, y, z=3):
    pass


@fixture(scope="module")
def cached_func(tmpdir_factory):
    # Create a Memory object to test decorated functions.
    # We should be careful not to call the decorated functions, so that
    # cache directories are not created in the temp dir.
    cachedir = tmpdir_factory.mktemp("joblib_test_func_inspect")
    mem = Memory(cachedir.strpath)

    @mem.cache
    def cached_func_inner(x):
        return x

    return cached_func_inner


class Klass(object):
    def f(self, x):
        return x


###############################################################################
# Tests


@parametrize(
    "func,args,filtered_args",
    [
        # Edge case: negative, zero, float
        (f, [[], (0.0,)], {"x": 0.0, "y": 0}),
        (f, [["x"], (-999,)], {"y": 0}),
        (f, [["y"], (100000,), {"y": -1}], {"x": 100000}),
        (f, [["y"], (10,), {"y": 50}], {"x": 10}),  # <--- FIXED OUTPUT HERE
        (f, [["x", "y"], (7,)], {}),
        (f, [[], (0,), {"y": 0}], {"x": 0, "y": 0}),
        (f, [["y"], (), {"x": 3.14, "y": -100}], {"x": 3.14}),
        (g, [[], (), {"x": 0}], {"x": 0}),
        (i, [[], (-42,)], {"x": -42}),
    ],
)
def test_filter_args(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args


def test_filter_args_method():
    obj = Klass()
    # Edge case: pass negative number
    assert filter_args(obj.f, [], (-55,)) == {"x": -55, "self": obj}


@parametrize(
    "func,args,filtered_args",
    [
        # Edge cases
        (h, [[], (0,)], {"x": 0, "y": 0, "*": [], "**": {}}),
        (h, [[], (1, 2, -7, 8)], {"x": 1, "y": 2, "*": [-7, 8], "**": {}}),
        (h, [[], (-1, 25), {"ee": -3}], {"x": -1, "y": 25, "*": [], "**": {"ee": -3}}),
        (h, [["*"], (2.5, -6, 25), {"ee": 77}], {"x": 2.5, "y": -6, "**": {"ee": 77}}),
    ],
)
def test_filter_varargs(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args


test_filter_kwargs_extra_params = [
    (m1, [[], (-1,), {"y": None}], {"x": -1, "y": None}),
    (m2, [[], (1000,), {"y": 555}], {"x": 1000, "y": 555, "z": 3}),
]


@parametrize(
    "func,args,filtered_args",
    [
        # Edge cases for k
        (k, [[], (), {"foo": None}], {"*": [], "**": {"foo": None}}),
        (k, [[], (999999, -42)], {"*": [999999, -42], "**": {}}),
    ]
    + test_filter_kwargs_extra_params,
)
def test_filter_kwargs(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args


def test_filter_args_2():
    # Edge case: large numbers
    assert filter_args(j, [], (1000, -2000), {"ee": 99}) == {"x": 1000, "y": -2000, "**": {"ee": 99}}

    ff = functools.partial(f, 0)
    # filter_args has to special-case partial
    assert filter_args(ff, [], (3000,)) == {"*": [3000], "**": {}}
    assert filter_args(ff, ["y"], (-1,)) == {"*": [-1], "**": {}}


@parametrize("func,funcname", [(f, "f"), (g, "g"), (cached_func, "cached_func")])
def test_func_name(func, funcname):
    # Check that we are not confused by decoration
    # here testcase 'cached_func' is the function itself
    assert get_func_name(func)[1] == funcname


def test_func_name_on_inner_func(cached_func):
    # Check that we are not confused by decoration
    # here testcase 'cached_func' is the 'cached_func_inner' function
    # returned by 'cached_func' fixture
    assert get_func_name(cached_func)[1] == "cached_func_inner"


def test_func_name_collision_on_inner_func():
    # Check that two functions defining and caching an inner function
    # with the same do not cause (module, name) collision
    def f():
        def inner_func():
            return  # pragma: no cover

        return get_func_name(inner_func)

    def g():
        def inner_func():
            return  # pragma: no cover

        return get_func_name(inner_func)

    module, name = f()
    other_module, other_name = g()

    assert name == other_name
    assert module != other_module


def test_func_inspect_errors():
    # Check that func_inspect is robust and will work on weird objects
    # New string function: upper
    assert get_func_name("a".upper)[-1] == "upper"
    assert get_func_code("a".upper)[1:] == (None, -1)
    # Test a non-lambda callable, e.g., int
    assert get_func_name(int, win_characters=False)[-1] == "int"
    assert get_func_code(int)[1] is None
    ff = lambda x: x  # noqa: E731
    assert get_func_name(ff, win_characters=False)[-1] == "<lambda>"
    assert get_func_code(ff)[1] == __file__.replace(".pyc", ".py")
    # Simulate a function defined in __main__
    ff.__module__ = "__main__"
    assert get_func_name(ff, win_characters=False)[-1] == "<lambda>"
    assert get_func_code(ff)[1] == __file__.replace(".pyc", ".py")


def func_with_kwonly_args(a, b, *, kw1="kw1", kw2="kw2"):
    pass


def func_with_signature(a: int, b: int) -> None:
    pass


def test_filter_args_edge_cases():
    # edge cases: kw1 is None, kw2 negative
    assert filter_args(func_with_kwonly_args, [], (0, -1), {"kw1": None, "kw2": -999}) == {
        "a": 0,
        "b": -1,
        "kw1": None,
        "kw2": -999,
    }

    # filter_args doesn't care about keyword-only arguments so you
    # can pass 'kw1' into *args without any problem
    with raises(ValueError) as excinfo:
        filter_args(func_with_kwonly_args, [], (20, 21, 22), {"kw2": 100})
    excinfo.match("Keyword-only parameter 'kw1' was passed as positional parameter")

    # edge case: ignore 'kw2', pass very large and zero values
    assert filter_args(
        func_with_kwonly_args, ["b", "kw2"], (12345678, 0), {"kw1": 314, "kw2": 0}
    ) == {"a": 12345678, "kw1": 314}

    # edge case: filter_args with floats and missing second param
    assert filter_args(func_with_signature, ["b"], (0.0, 999999.999)) == {"a": 0.0}


def test_bound_methods():
    """Make sure that calling the same method on two different instances
    of the same class does resolv to different signatures.
    """
    a = Klass()
    b = Klass()
    assert filter_args(a.f, [], (1,)) != filter_args(b.f, [], (1,))


@parametrize(
    "exception,regex,func,args",
    [
        (
            ValueError,
            "ignore_lst must be a list of parameters to ignore",
            f,
            ["bar", (None,)],
        ),
        (
            ValueError,
            r"Ignore list: argument \'(.*)\' is not defined",
            g,
            [["bar"], (None,)],
        ),
        (ValueError, "Wrong number of arguments", h, [[]]),
    ],
)
def test_filter_args_error_msg(exception, regex, func, args):
    """Make sure that filter_args returns decent error messages, for the
    sake of the user.
    """
    with raises(exception) as excinfo:
        filter_args(func, *args)
    excinfo.match(regex)


def test_filter_args_no_kwargs_mutation():
    """None-regression test against 0.12.0 changes.

    https://github.com/joblib/joblib/pull/75

    Make sure filter args doesn't mutate the kwargs dict that gets passed in.
    """
    kwargs = {"x": 0}
    filter_args(g, [], [], kwargs)
    assert kwargs == {"x": 0}


def test_clean_win_chars():
    string = r"C:\foo\bar\main.py"
    mangled_string = _clean_win_chars(string)
    for char in ("\\", ":", "<", ">", "!"):
        assert char not in mangled_string


@parametrize(
    "func,args,kwargs,sgn_expected",
    [
        (g, [list(range(5))], {}, "g([0, 1, 2, 3, 4])"),
        (k, [1, 2, (3, 4)], {"y": True}, "k(1, 2, (3, 4), y=True)"),
    ],
)
def test_format_signature(func, args, kwargs, sgn_expected):
    # Test signature formatting.
    path, sgn_result = format_signature(func, *args, **kwargs)
    assert sgn_result == sgn_expected


def test_format_signature_long_arguments():
    shortening_threshold = 1500
    # shortening gets it down to 700 characters but there is the name
    # of the function in the signature and a few additional things
    # like dots for the ellipsis
    shortening_target = 700 + 10

    arg = "a" * shortening_threshold
    _, signature = format_signature(h, arg)
    assert len(signature) < shortening_target

    nb_args = 5
    args = [arg for _ in range(nb_args)]
    _, signature = format_signature(h, *args)
    assert len(signature) < shortening_target * nb_args

    kwargs = {str(i): arg for i, arg in enumerate(args)}
    _, signature = format_signature(h, **kwargs)
    assert len(signature) < shortening_target * nb_args

    _, signature = format_signature(h, *args, **kwargs)
    assert len(signature) < shortening_target * 2 * nb_args


@with_numpy
def test_format_signature_numpy():
    """Test the format signature formatting with numpy."""


def test_special_source_encoding():
    from joblib.test.test_func_inspect_special_encoding import big5_f

    func_code, source_file, first_line = get_func_code(big5_f)
    assert first_line == 5
    assert "def big5_f():" in func_code
    assert "test_func_inspect_special_encoding" in source_file


def _get_code():
    from joblib.test.test_func_inspect_special_encoding import big5_f

    return get_func_code(big5_f)[0]


def test_func_code_consistency():
    from joblib.parallel import Parallel, delayed

    codes = Parallel(n_jobs=2)(delayed(_get_code)() for _ in range(5))
    assert len(set(codes)) == 1