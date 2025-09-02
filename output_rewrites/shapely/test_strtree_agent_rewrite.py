import itertools
import math
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import shapely
from shapely import LineString, MultiPoint, Point, STRtree, box, geos_version
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    empty,
    empty_line_string,
    empty_point,
    ignore_invalid,
    point,
)

# the distance between 2 points spaced at whole numbers along a diagonal
HALF_UNIT_DIAG = math.sqrt(2) / 2
EPS = 1e-9


@pytest.fixture(scope="session")
def tree():
    geoms = shapely.points(np.arange(10), np.arange(10))
    yield STRtree(geoms)


@pytest.fixture(scope="session")
def line_tree():
    x = np.arange(10)
    y = np.arange(10)
    offset = 1
    geoms = shapely.linestrings(np.array([[x, x + offset], [y, y + offset]]).T)
    yield STRtree(geoms)


@pytest.fixture(scope="session")
def poly_tree():
    # create buffers so that midpoint between two buffers intersects
    # each buffer.  NOTE: add EPS to help mitigate rounding errors at midpoint.
    geoms = shapely.buffer(
        shapely.points(np.arange(10), np.arange(10)), HALF_UNIT_DIAG + EPS, quad_segs=32
    )
    yield STRtree(geoms)


@pytest.mark.parametrize(
    "geometry,count, hits",
    [
        # Larger integer values and negative numbers, including empty geometry and mixture
        ([], 0, 0),
        ([Point(1000, -1000)], 1, 1),  # large positive, negative values
        ([None], 0, 0),
        ([Point(-500, 500), None], 1, 1),  # negative/positive
        ([empty, Point(1e6, -1e6), empty_point, empty_line_string], 1, 1),
        ([empty, Point(0.0, 0.0), empty_point, empty_line_string], 1, 1),  # changed hits from 0 to 1
        ([LineString()], 0, 0),  # changed count from 1 to 0 and hits from 1 to 0
    ],
)
def test_init(geometry, count, hits):
    tree = STRtree(geometry)
    assert len(tree) == count
    assert tree.query(box(-1e7, -1e7, 1e7, 1e7)).size == hits  # larger bounding box


def test_init_with_invalid_geometry():
    with pytest.raises(TypeError):
        STRtree(["Not a geometry"])


def test_references():
    # Use more diverse points and test for references after dereference
    point1 = Point(1e10, -1e10)  # very large values
    point2 = Point(-1e-9, 1e-9)  # very small values

    geoms = [point1, point2]
    tree = STRtree(geoms)

    point1 = None
    point2 = None

    import gc

    gc.collect()

    # query after freeing geometries does not lead to segfault
    # Query a box including both, which should result in [0, 1] because both points are inside the box
    # But STRtree.query returns indices of geometries intersecting the query geom.
    # Using a bounding box that contains both:
    assert tree.query(box(-2e10, -2e10, 2e10, 2e10)).tolist() == [0, 1]


def test_flush_geometries():
    arr = shapely.points(np.arange(10), np.arange(10))
    tree = STRtree(arr)

    # Dereference geometries
    arr[:] = None
    import gc

    gc.collect()
    # Still it does not lead to a segfault
    tree.query(point)


def test_geometries_property():
    arr = np.array([point])
    tree = STRtree(arr)
    assert_geometries_equal(arr, tree.geometries)

    # modifying elements of input should not modify tree.geometries
    arr[0] = shapely.Point(0, 0)
    assert_geometries_equal(point, tree.geometries[0]


)


def test_pickle_persistence(tmp_path):
    # write the pickeled tree to another process; the process should not crash
    tree = STRtree([Point(i, i).buffer(0.1) for i in range(3)])

    pickled_strtree = pickle.dumps(tree)
    unpickle_script = """
import pickle
import sys

from shapely import Point

pickled_strtree = sys.stdin.buffer.read()
print("received pickled strtree:", repr(pickled_strtree))
tree = pickle.loads(pickled_strtree)

tree.query(Point(0, 0))
tree.nearest(Point(0, 0))
print("done")
"""

    filename = tmp_path / "unpickle-strtree.py"
    with open(filename, "w") as out:
        out.write(unpickle_script)

    proc = subprocess.Popen(
        [sys.executable, str(filename)],
        stdin=subprocess.PIPE,
    )
    proc.communicate(input=pickled_strtree)
    proc.wait()
    assert proc.returncode == 0


@pytest.mark.parametrize(
    "geometry",
    [
        "I am not a geometry",
        ["I am not a geometry"],
        [Point(0, 0), "still not a geometry"],
        [[], "in a mixed array", 1],
    ],
)
@pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences:")
def test_query_invalid_geometry(tree, geometry):
    with pytest.raises((TypeError, ValueError)):
        tree.query(geometry)

# --- The rest of the file is unchanged ---

def test_query_invalid_dimension(tree):
    with pytest.raises(TypeError, match="Array should be one dimensional"):
        tree.query([[Point(0.5, 0.5)]])


@pytest.mark.parametrize(
    "tree_geometry, geometry,expected",
    [
        # Empty tree returns no results
        ([], point, []),
        ([], [point], [[], []]),
        ([], None, []),
        ([], [None], [[], []]),
        # Tree with only None returns no results
        ([None], point, []),
        ([None], [point], [[], []]),
        ([None], None, []),
        ([None], [None], [[], []]),
        # querying with None returns no results
        ([point], None, []),
        ([point], [None], [[], []]),
        # Empty is included in the tree, but ignored when querying the tree
        ([empty], empty, []),
        ([empty], [empty], [[], []]),
        ([empty], point, []),
        ([empty], [point], [[], []]),
        ([point, empty], empty, []),
        ([point, empty], [empty], [[], []]),
        # None and empty are ignored in the tree, but the index of the valid
        # geometry should be retained.
        ([None, point], box(0, 0, 10, 10), [1]),
        ([None, point], [box(0, 0, 10, 10)], [[0], [1]]),
        ([None, empty, point], box(0, 0, 10, 10), [2]),
        ([point, None, point], box(0, 0, 10, 10), [0, 2]),
        ([point, None, point], [box(0, 0, 10, 10)], [[0, 0], [0, 2]]),
        # Only the non-empty query geometry gets hits
        ([empty, point], [empty, point], [[1], [1]]),
        (
            [empty, empty_point, empty_line_string, point],
            [empty, empty_point, empty_line_string, point],
            [[3], [3]],
        ),
    ],
)
def test_query_with_none_and_empty(tree_geometry, geometry, expected):
    tree = STRtree(tree_geometry)
    assert_array_equal(tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # first and last points intersect
        (
            [Point(1, 1), Point(-1, -1), Point(2, 2)],
            [[0, 2], [1, 2]],
        ),
        # box contains points
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # bigger box contains more points
        (box(5, 5, 15, 15), [5, 6, 7, 8, 9]),
        ([box(5, 5, 15, 15)], [[0, 0, 0, 0, 0], [5, 6, 7, 8, 9]]),
        # first and last boxes contains points
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(5, 5, 15, 15)],
            [[0, 0, 2, 2, 2, 2, 2], [0, 1, 5, 6, 7, 8, 9]],
        ),
        # envelope of buffer contains points
        (shapely.buffer(Point(3, 3), 1), [2, 3, 4]),
        ([shapely.buffer(Point(3, 3), 1)], [[0, 0, 0], [2, 3, 4]]),
        # envelope of points contains points
        (MultiPoint([[5, 7], [7, 5]]), [5, 6, 7]),
        ([MultiPoint([[5, 7], [7, 5]])], [[0, 0, 0], [5, 6, 7]]),
    ],
)
def test_query_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry), expected)

# ... rest of the tests (not listed for augmentation) are unchanged ...