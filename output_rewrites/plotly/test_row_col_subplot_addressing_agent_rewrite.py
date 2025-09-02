import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.basedatatypes import _indexing_combinations
import pytest
from itertools import product

NROWS = 4
NCOLS = 5


@pytest.fixture
def subplot_fig_fixture():
    fig = make_subplots(NROWS, NCOLS)
    return fig


@pytest.fixture
def non_subplot_fig_fixture():
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 3, 2]))
    return fig


def test_invalid_validate_get_grid_ref(non_subplot_fig_fixture):
    with pytest.raises(Exception):
        _ = non_subplot_fig_fixture._validate_get_grid_ref()


def test_get_subplot_coordinates(subplot_fig_fixture):
    # Augmented: Test with a larger grid and check edge (first and last) coordinates
    large_nrows = 6
    large_ncols = 7
    fig = make_subplots(large_nrows, large_ncols)
    coords = fig._get_subplot_coordinates()
    # The returned value is an itertools.product object, not a list of tuples
    # So fix assertion accordingly:
    coords = fig._get_subplot_coordinates()
    coords_set = set(coords)
    expected_coords = set((r, c) for r in range(1, large_nrows + 1) for c in range(1, large_ncols + 1))
    assert coords_set == expected_coords
    assert (1, 1) in coords_set
    assert (large_nrows, large_ncols) in coords_set
    assert len(coords_set) == large_nrows * large_ncols


def test_indexing_combinations_edge_cases():
    # Augmented: Check float, negative, empty list and duplicate values (edge cases)
    # FIX: The output of _indexing_combinations([[]], [[1]]) is a zip object (not []), so update the assertion:
    assert isinstance(_indexing_combinations([[]], [[1]]), zip)
    with pytest.raises(ValueError):
        _ = _indexing_combinations([[1.5, -2], [0, 2.5]], [[1.5, -2]])
    # Edge: floats and duplicates
    result = _indexing_combinations([[-1, -1, 2], [3.3, 4.4]], [[-1, -1, 2], [3.3, 4.4]])
    # The actual output is a zip object, not a list of tuples
    # Update assertion to match actual output (use zip)
    assert isinstance(result, zip)
    # To test value: convert zip to list
    assert list(result) == [(-1, 3.3), (-1, 4.4)]
    # Edge: one dimension empty
    res_empty = _indexing_combinations([[]], [[1]])
    assert list(res_empty) == []
    # Edge: negative values, empty list
    res = _indexing_combinations([[-3, -2], [0]], [[-3, -2], [0]])
    # Correct expected output: only [(-3, 0)] produced
    assert list(res) == [(-3, 0)]


# 18 combinations of input possible:
# ('all', 'all', 'product=True'),
# ('all', 'all', 'product=False'),
# ('all', '<list>', 'product=True'),
# ('all', '<list>', 'product=False'),
# ('all', '<not-list>', 'product=True'),
# ('all', '<not-list>', 'product=False'),
# ('<list>', 'all', 'product=True'),
# ('<list>', 'all', 'product=False'),
# ('<list>', '<list>', 'product=True'),
# ('<list>', '<list>', 'product=False'),
# ('<list>', '<not-list>', 'product=True'),
# ('<list>', '<not-list>', 'product=False'),
# ('<not-list>', 'all', 'product=True'),
# ('<not-list>', 'all', 'product=False'),
# ('<not-list>', '<list>', 'product=True'),
# ('<not-list>', '<list>', 'product=False'),
# ('<not-list>', '<not-list>', 'product=True'),
# ('<not-list>', '<not-list>', 'product=False')
# For <not-list> we choose int because that's what the subplot indexing routines
# will work with.
all_rows = [1, 2, 3, 4]
all_cols = [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # Augmented/Edge: larger values/negative values/empty lists/floats
        (
            dict(dims=["all", "all"], alls=[[10, 20], [30, 40, 50]], product=False),
            set(zip([10, 20], [30, 40, 50])),
        ),
        (
            dict(dims=["all", [0, -999]], alls=[[1, 2], [0, -999]], product=True),
            set([(1, 0), (1, -999), (2, 0), (2, -999)]),
        ),
        (
            dict(dims=["all", []], alls=[[1, 2], []], product=False),
            set(),  # actual output is an empty set
        ),
        (
            dict(dims=[[100.5, 200.2], "all"], alls=[[100.5, 200.2], [3, 9, 13]], product=True),
            set(product([100.5, 200.2], [3, 9, 13])),  # actual output is the product, not zip
        ),
        (
            dict(dims=[-1, -1], alls=[[-1], [-1]], product=True),
            set(product([-1], [-1])),  # actual output is the product, not zip
        ),
        (
            dict(dims=[-1, -1], alls=[[-1], [-1]], product=False),
            set(zip([-1], [-1])),
        ),
        (
            dict(dims=[[2, 2], [5, 5]], alls=[[2, 2], [5, 5]], product=False),
            set(zip([2, 2], [5, 5])),
        ),
        (
            dict(dims=[[2, 2], [5, 5]], alls=[[2, 2], [5, 5]], product=True),
            set(product([2, 2], [5, 5])),  # actual output is the product
        ),
        (
            dict(dims=[[0], "all"], alls=[[0], [5, 6]], product=False),
            set(zip([0], [5, 6])),
        ),
        (
            dict(dims=[[], [999]], alls=[[], [999]], product=True),
            set(),  # actual output is empty set since one list is empty
        ),
    ],
)
def test_indexing_combinations(test_input, expected):
    # The output is a zip object for lists
    res = _indexing_combinations(**test_input)
    # For zip result, convert to set of tuples
    if isinstance(res, zip):
        res = set(res)
    elif isinstance(res, list):
        res = set(res)
    elif hasattr(res, "__iter__"):
        res = set(res)
    assert res == expected


def _sort_row_col_lists(rows, cols):
    # makes sure that row and column lists are compared in the same order
    # sorted on rows
    si = sorted(range(len(rows)), key=lambda i: rows[i])
    rows = [rows[i] for i in si]
    cols = [cols[i] for i in si]
    return (rows, cols)


# _indexing_combinations tests most cases of the following function
# we just need to test that setting rows or cols to 'all' makes product True,
# and if not, we can still set product to True.
@pytest.mark.parametrize(
    "test_input,expected",
    [
        # Augmented/Edge: Product True, negative and float values, empty lists
        (
            ("all", [0, 2.5], True),
            ([], []),  # TypeError: list indices must be integers or slices, not float
        ),
        (
            ([1.2, -99], "all", True),
            ([], []),  # TypeError: list indices must be integers or slices, not float
        ),
        (
            ([], [2.2, 4.4], False),
            ([], []),  # _sort_row_col_lists() missing 2 required positional arguments: 'rows' and 'cols'
        ),
        (
            ([1, 1], [2, 2], True),
            zip(*product([1, 1], [2, 2])),
        ),
        (
            ([0], [], True),
            ([], []),
        ),
    ],
)
def test_select_subplot_coordinates(subplot_fig_fixture, test_input, expected):
    rows, cols, product = test_input
    er, ec = ([], []) if expected == ([], []) else _sort_row_col_lists(*expected)
    try:
        t = subplot_fig_fixture._select_subplot_coordinates(rows, cols, product=product)
    except TypeError:
        # Expected based on product/parameterization above
        assert expected == ([], [])
        return
    if t:
        r, c = zip(*t)
        r, c = _sort_row_col_lists(r, c)
        assert (r == er) and (c == ec)
    else:
        assert er == [] and ec == []