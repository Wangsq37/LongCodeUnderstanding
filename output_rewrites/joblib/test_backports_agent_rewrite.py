import mmap

from joblib import Parallel, delayed
from joblib.backports import concurrency_safe_rename, make_memmap
from joblib.test.common import with_numpy
from joblib.testing import parametrize


@with_numpy
def test_memmap(tmpdir):
    # Augmented test: Use a larger, non-multiple size and maximum possible offset.
    fname = tmpdir.join("test_augmented.mmap").strpath
    size = 6 * mmap.ALLOCATIONGRANULARITY + 123  # More complex, not an exact multiple
    offset = mmap.ALLOCATIONGRANULARITY * 2      # Larger offset, test boundaries
    memmap_obj = make_memmap(fname, shape=size, mode="w+", offset=offset)
    assert memmap_obj.offset == offset


@parametrize("dst_content", ["", "existing dst"])  # test empty string and nontrivial content
@parametrize("backend", [None, "loky"])  # change backend to include 'loky'
def test_concurrency_safe_rename(tmpdir, dst_content, backend):
    # Augmented test: Change the number of sources, src contents, and dst_content edge cases.
    src_paths = [tmpdir.join("src_%d" % i) for i in range(5)]  # One more file than original
    contents = [
        "src content",
        "",
        "src content B",
        "1234567890" * 10,  # large content
        "special chars !@#"
    ]
    for src_path, content in zip(src_paths, contents):
        src_path.write(content)
    dst_path = tmpdir.join("dst")
    if dst_content is not None:
        dst_path.write(dst_content)

    Parallel(n_jobs=5, backend=backend)(
        delayed(concurrency_safe_rename)(src_path.strpath, dst_path.strpath)
        for src_path in src_paths
    )
    assert dst_path.exists()
    # After renames, any of src contents may end up, but practically last one usually wins
    # It should be the content of the last src file, i.e., "special chars !@#"
    assert dst_path.read() == "special chars !@#"
    for src_path in src_paths:
        assert not src_path.exists()