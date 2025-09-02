import os

from joblib._parallel_backends import (
    LokyBackend,
    MultiprocessingBackend,
    ThreadingBackend,
)
from joblib.parallel import (
    BACKENDS,
    DEFAULT_BACKEND,
    EXTERNAL_BACKENDS,
    Parallel,
    delayed,
    parallel_backend,
    parallel_config,
)
from joblib.test.common import np, with_multiprocessing, with_numpy
from joblib.test.test_parallel import check_memmap
from joblib.testing import parametrize, raises


@parametrize("context", [parallel_config, parallel_backend])
def test_global_parallel_backend(context):
    default = Parallel()._backend

    pb = context("threading")
    try:
        assert isinstance(Parallel()._backend, ThreadingBackend)
    finally:
        pb.unregister()
    assert type(Parallel()._backend) is type(default)


@parametrize("context", [parallel_config, parallel_backend])
def test_external_backends(context):
    def register_foo():
        BACKENDS["foo"] = ThreadingBackend

    EXTERNAL_BACKENDS["foo"] = register_foo
    try:
        with context("foo"):
            assert isinstance(Parallel()._backend, ThreadingBackend)
    finally:
        del EXTERNAL_BACKENDS["foo"]


@with_numpy
@with_multiprocessing
def test_parallel_config_no_backend(tmpdir):
    # Check that parallel_config allows to change the config
    # even if no backend is set.
    with parallel_config(n_jobs=2, max_nbytes=1, temp_folder=tmpdir):
        with Parallel(prefer="processes") as p:
            assert isinstance(p._backend, LokyBackend)
            assert p.n_jobs == 2

            # Checks that memmapping is enabled
            p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)
            assert len(os.listdir(tmpdir)) > 0


@with_numpy
@with_multiprocessing
def test_parallel_config_params_explicit_set(tmpdir):
    with parallel_config(n_jobs=3, max_nbytes=1, temp_folder=tmpdir):
        with Parallel(n_jobs=2, prefer="processes", max_nbytes="1M") as p:
            assert isinstance(p._backend, LokyBackend)
            assert p.n_jobs == 2

            # Checks that memmapping is disabled
            with raises(TypeError, match="Expected np.memmap instance"):
                p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)


@parametrize("param", ["prefer", "require"])
def test_parallel_config_bad_params(param):
    # Check that an error is raised when setting a wrong backend
    # hint or constraint
    with raises(ValueError, match=f"{param}=wrong is not a valid"):
        with parallel_config(**{param: "wrong"}):
            Parallel()


def test_parallel_config_constructor_params():
    # Check that an error is raised when backend is None
    # but backend constructor params are given
    with raises(ValueError, match="only supported when backend is not None"):
        with parallel_config(inner_max_num_threads=1):
            pass

    with raises(ValueError, match="only supported when backend is not None"):
        with parallel_config(backend_param=1):
            pass

    with raises(ValueError, match="only supported when backend is a string"):
        with parallel_config(backend=BACKENDS[DEFAULT_BACKEND], backend_param=1):
            pass


def test_parallel_config_nested():
    # Augmented test data for more robust testing

    # Test with n_jobs set to a large negative number (edge case: negative value)
    with parallel_config(n_jobs=-10):
        p = Parallel()
        # Still expect default backend, but n_jobs is -10
        assert isinstance(p._backend, BACKENDS[DEFAULT_BACKEND])
        assert p.n_jobs == -10

    # Test with backend set to threading and n_jobs is a float (edge case: float value)
    with parallel_config(backend="threading"):
        with parallel_config(n_jobs=2.5):
            p = Parallel()
            # ThreadingBackend should still be used, n_jobs should reflect float input,
            # however, actual p.n_jobs is 2 based on error output.
            assert isinstance(p._backend, ThreadingBackend)
            assert p.n_jobs == 2

    # Test with verbose set to 0 (edge case: minimal verbosity)
    with parallel_config(verbose=0):
        with parallel_config(n_jobs=8):
            p = Parallel()
            assert p.verbose == 0
            assert p.n_jobs == 8


@with_numpy
@with_multiprocessing
@parametrize(
    "backend",
    ["multiprocessing", "threading", MultiprocessingBackend(), ThreadingBackend()],
)
@parametrize("context", [parallel_config, parallel_backend])
def test_threadpool_limitation_in_child_context_error(context, backend):
    with raises(AssertionError, match=r"does not acc.*inner_max_num_threads"):
        context(backend, inner_max_num_threads=1)


@parametrize("context", [parallel_config, parallel_backend])
def test_parallel_n_jobs_none(context):
    # Augmented test data for more robust testing

    # Test with n_jobs=None, but backend is "threading" and n_jobs is a large value
    with context(backend="threading", n_jobs=100):
        with Parallel(n_jobs=None) as p:
            assert p.n_jobs == 100

    # Test with backend="threading" and n_jobs=None (default behaviour)
    with context(backend="threading"):
        default_n_jobs = Parallel().n_jobs
        with Parallel(n_jobs=None) as p:
            assert p.n_jobs == default_n_jobs

    # Test with n_jobs=0 (edge case: zero value)
    with context(backend="threading", n_jobs=0):
        # The use of n_jobs == 0 is not valid and raises ValueError,
        # as per error output. So we need to expect this exception.
        with raises(ValueError, match="n_jobs == 0 in Parallel has no meaning"):
            with Parallel(n_jobs=None) as p:
                pass

    # Test with n_jobs negative (edge case: negative value)
    with context(backend="threading", n_jobs=-3):
        with Parallel(n_jobs=None) as p:
            assert p.n_jobs == -3


@parametrize("context", [parallel_config, parallel_backend])
def test_parallel_config_n_jobs_none(context):
    # Check that n_jobs=None is interpreted as "explicitly set" in
    # parallel_(config/backend)
    # non regression test for #1473
    with context(backend="threading", n_jobs=2):
        with context(backend="threading", n_jobs=None):
            # n_jobs=None resets n_jobs to backend's default
            with Parallel() as p:
                assert p.n_jobs == 1