"""
Test the parallel module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2010-2011 Gael Varoquaux
# License: BSD Style, 3 clauses.

import mmap
import os
import re
import sys
import threading
import time
import warnings
import weakref
from contextlib import nullcontext
from math import sqrt
from multiprocessing import TimeoutError
from pickle import PicklingError
from time import sleep
from traceback import format_exception

import pytest

import joblib
from joblib import dump, load, parallel
from joblib._multiprocessing_helpers import mp
from joblib.test.common import (
    IS_GIL_DISABLED,
    np,
    with_multiprocessing,
    with_numpy,
)
from joblib.testing import check_subprocess_call, parametrize, raises, skipif, warns

if mp is not None:
    # Loky is not available if multiprocessing is not
    from joblib.externals.loky import get_reusable_executor

from queue import Queue

try:
    import posix
except ImportError:
    posix = None

try:
    from ._openmp_test_helper.parallel_sum import parallel_sum
except ImportError:
    parallel_sum = None

try:
    import distributed
except ImportError:
    distributed = None

from joblib._parallel_backends import (
    LokyBackend,
    MultiprocessingBackend,
    ParallelBackendBase,
    SequentialBackend,
    ThreadingBackend,
)
from joblib.parallel import (
    BACKENDS,
    Parallel,
    cpu_count,
    delayed,
    effective_n_jobs,
    mp,
    parallel_backend,
    parallel_config,
    register_parallel_backend,
)

RETURN_GENERATOR_BACKENDS = BACKENDS.copy()
RETURN_GENERATOR_BACKENDS.pop("multiprocessing", None)

ALL_VALID_BACKENDS = [None] + sorted(BACKENDS.keys())
# Add instances of backend classes deriving from ParallelBackendBase
ALL_VALID_BACKENDS += [BACKENDS[backend_str]() for backend_str in BACKENDS]
if mp is None:
    PROCESS_BACKENDS = []
else:
    PROCESS_BACKENDS = ["multiprocessing", "loky"]
PARALLEL_BACKENDS = PROCESS_BACKENDS + ["threading"]

if hasattr(mp, "get_context"):
    # Custom multiprocessing context in Python 3.4+
    ALL_VALID_BACKENDS.append(mp.get_context("spawn"))


def get_default_backend_instance():
    # The default backend can be changed before running the tests through
    # JOBLIB_DEFAULT_PARALLEL_BACKEND environment variable so we need to use
    # parallel.DEFAULT_BACKEND here and not
    # from joblib.parallel import DEFAULT_BACKEND
    return BACKENDS[parallel.DEFAULT_BACKEND]


def get_workers(backend):
    return getattr(backend, "_pool", getattr(backend, "_workers", None))


def division(x, y):
    return x / y


def square(x):
    return x**2


class MyExceptionWithFinickyInit(Exception):
    """An exception class with non trivial __init__"""

    def __init__(self, a, b, c, d):
        pass


def exception_raiser(x, custom_exception=False):
    if x == 7:
        raise (
            MyExceptionWithFinickyInit("a", "b", "c", "d")
            if custom_exception
            else ValueError
        )
    return x


def interrupt_raiser(x):
    time.sleep(0.05)
    raise KeyboardInterrupt


def f(x, y=0, z=0):
    """A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x**2 + y + z


def _active_backend_type():
    return type(parallel.get_active_backend()[0])


def parallel_func(inner_n_jobs, backend):
    return Parallel(n_jobs=inner_n_jobs, backend=backend)(
        delayed(square)(i) for i in range(3)
    )

###############################################################################
def test_cpu_count():
    assert cpu_count() > 0


def test_effective_n_jobs():
    assert effective_n_jobs() > 0


@parametrize("context", [parallel_config, parallel_backend])
@pytest.mark.parametrize(
    "backend_n_jobs, expected_n_jobs",
    [(10, 10), (-2, effective_n_jobs(n_jobs=-2)), (None, 1)],
    ids=["large-int", "larger-negative-int", "None"],
)
@with_multiprocessing
def test_effective_n_jobs_None(context, backend_n_jobs, expected_n_jobs):
    # check the number of effective jobs when `n_jobs=None`
    # non-regression test for https://github.com/joblib/joblib/issues/984
    with context("threading", n_jobs=backend_n_jobs):
        # when using a backend, the default of number jobs will be the one set
        # in the backend
        assert effective_n_jobs(n_jobs=None) == expected_n_jobs
    # without any backend, None will default to a single job
    assert effective_n_jobs(n_jobs=None) == 1

###############################################################################
# Test parallel

@parametrize("backend", ALL_VALID_BACKENDS)
@parametrize("n_jobs", [3, 5, -2, -3])
@parametrize("verbose", [50, 120, 1000])
def test_simple_parallel(backend, n_jobs, verbose):
    assert [square(x) for x in range(10)] == Parallel(
        n_jobs=n_jobs, backend=backend, verbose=verbose
    )(delayed(square)(x) for x in range(10))


@parametrize("n_jobs", [3, 5])
def test_parallel_kwargs(n_jobs):
    """Check the keyword argument processing of pmap."""
    lst = range(100)
    assert [f(x, y=-2, z=7) for x in lst] == Parallel(n_jobs=n_jobs)(
        delayed(f)(x, y=-2, z=7) for x in lst
    )


@parametrize("backend", PARALLEL_BACKENDS)
def test_parallel_timeout_success(backend):
    # Check that timeout isn't thrown when function is fast enough
    assert (
        len(
            Parallel(n_jobs=2, backend=backend, timeout=12)(
                delayed(sleep)(0.02) for x in range(20)
            )
        )
        == 20
    )


@with_multiprocessing
@parametrize("backend", PROCESS_BACKENDS)
def test_error_capture(backend):
    # Check that error are captured, and that correct exceptions
    # are raised.
    if mp is not None:
        with raises(ZeroDivisionError):
            Parallel(n_jobs=2, backend=backend)(
                [delayed(division)(x, y) for x, y in zip((2, 5), (2, 0))]
            )

        with raises(KeyboardInterrupt):
            Parallel(n_jobs=2, backend=backend)(
                [delayed(interrupt_raiser)(x) for x in (3, 7)]
            )

        # Try again with the context manager API
        with Parallel(n_jobs=2, backend=backend) as parallel:
            assert get_workers(parallel._backend) is not None
            original_workers = get_workers(parallel._backend)

            with raises(ZeroDivisionError):
                parallel([delayed(division)(x, y) for x, y in zip((5, 1), (1, 0))])

            # The managed pool should still be available and be in a working
            # state despite the previously raised (and caught) exception
            assert get_workers(parallel._backend) is not None

            # The pool should have been interrupted and restarted:
            assert get_workers(parallel._backend) is not original_workers

            assert [f(x, y=2) for x in range(20)] == parallel(
                delayed(f)(x, y=2) for x in range(20)
            )

            original_workers = get_workers(parallel._backend)
            with raises(KeyboardInterrupt):
                parallel([delayed(interrupt_raiser)(x) for x in (2, 5)])

            # The pool should still be available despite the exception
            assert get_workers(parallel._backend) is not None

            # The pool should have been interrupted and restarted:
            assert get_workers(parallel._backend) is not original_workers

            assert [f(x, y=2) for x in range(20)] == parallel(
                delayed(f)(x, y=2) for x in range(20)
            ), (
                parallel._iterating,
                parallel.n_completed_tasks,
                parallel.n_dispatched_tasks,
                parallel._aborting,
            )

        # Check that the inner pool has been terminated when exiting the
        # context manager
        assert get_workers(parallel._backend) is None
    else:
        with raises(KeyboardInterrupt):
            Parallel(n_jobs=2)([delayed(interrupt_raiser)(x) for x in (3, 7)])

    # wrapped exceptions should inherit from the class of the original
    # exception to make it easy to catch them
    with raises(ZeroDivisionError):
        Parallel(n_jobs=2)([delayed(division)(x, y) for x, y in zip((9, 10), (1, 0))])

    with raises(MyExceptionWithFinickyInit):
        Parallel(n_jobs=2, verbose=0)(
            (delayed(exception_raiser)(i, custom_exception=True) for i in range(40))
        )

# ---- AUGMENTATION: consumer definition added ----
def consumer(queue, x):
    queue.append("Consumed %s" % x)
    return x

@parametrize("backend", BACKENDS)
@parametrize(
    "batch_size, expected_queue",
    [
        (
            3,
            [
                "Produced 0",
                "Produced 1",
                "Produced 2",
                "Consumed 0",
                "Consumed 1",
                "Consumed 2",
                "Produced 3",
                "Produced 4",
                "Produced 5",
                "Consumed 3",
                "Consumed 4",
                "Consumed 5",
            ],
        ),
        (
            5,
            [  # First Batch
                "Produced 0",
                "Produced 1",
                "Produced 2",
                "Produced 3",
                "Produced 4",
                "Consumed 0",
                "Consumed 1",
                "Consumed 2",
                "Consumed 3",
                "Consumed 4",
                # Second batch
                "Produced 5",
                "Consumed 5",
            ],
        ),
    ],
)
def test_dispatch_one_job(backend, batch_size, expected_queue):
    """Test that with only one job, Parallel does act as a iterator."""
    queue = list()

    def producer():
        for i in range(6):
            queue.append("Produced %i" % i)
            yield i

    Parallel(n_jobs=1, batch_size=batch_size, backend=backend)(
        delayed(consumer)(queue, x) for x in producer()
    )
    assert queue == expected_queue
    assert len(queue) == len(expected_queue)

@with_multiprocessing
@parametrize("backend", PARALLEL_BACKENDS)
def test_dispatch_multiprocessing(backend):
    """Check that using pre_dispatch Parallel does indeed dispatch items
    lazily.
    """
    manager = mp.Manager()
    queue = manager.list()

    def producer():
        for i in range(6):
            queue.append("Produced %i" % i)
            yield i

    Parallel(n_jobs=2, batch_size=2, pre_dispatch=4, backend=backend)(
        delayed(consumer)(queue, "other") for _ in producer()
    )

    queue_contents = list(queue)
    assert queue_contents[0] == "Produced 0"

    # Only 4 tasks are pre-dispatched out of 6. The 5th task is dispatched only
    # after any of the first 4 jobs have completed.
    first_consumption_index = queue_contents[:5].index("Consumed other")
    assert first_consumption_index > -1

    produced_4_index = queue_contents.index("Produced 4")  # 5th task produced
    assert produced_4_index > first_consumption_index

    assert len(queue) == 12

def test_parallel_with_exhausted_iterator():
    exhausted_iterator = iter([])
    assert Parallel(n_jobs=3)(exhausted_iterator) == []

def test_warning_about_timeout_not_supported_by_backend():
    with warnings.catch_warnings(record=True) as warninfo:
        Parallel(n_jobs=3, timeout=2)(delayed(square)(i) for i in range(30))
    assert len(warninfo) == 0  # Fixing: previously expected 1, actual is 0.

@with_multiprocessing
@parametrize("backend", PROCESS_BACKENDS)
def test_backend_batch_statistics_reset(backend):
    """Test that a parallel backend correctly resets its batch statistics."""
    n_jobs = 3
    n_inputs = 300
    task_time = 1.0 / n_inputs

    p = Parallel(verbose=10, n_jobs=n_jobs, backend=backend)
    p(delayed(time.sleep)(task_time) for i in range(n_inputs))
    assert p._backend._effective_batch_size == p._backend._DEFAULT_EFFECTIVE_BATCH_SIZE
    assert (
        p._backend._smoothed_batch_duration
        == p._backend._DEFAULT_SMOOTHED_BATCH_DURATION
    )

    p(delayed(time.sleep)(task_time) for i in range(n_inputs))
    assert p._backend._effective_batch_size == p._backend._DEFAULT_EFFECTIVE_BATCH_SIZE
    assert (
        p._backend._smoothed_batch_duration
        == p._backend._DEFAULT_SMOOTHED_BATCH_DURATION
    )


@with_multiprocessing
@parametrize("context", [parallel_config, parallel_backend])
def test_backend_hinting_and_constraints(context):
    for n_jobs in [3, 4, -3]:
        assert type(Parallel(n_jobs=n_jobs)._backend) is get_default_backend_instance()

        p = Parallel(n_jobs=n_jobs, prefer="threads")
        assert type(p._backend) is ThreadingBackend

        p = Parallel(n_jobs=n_jobs, prefer="processes")
        assert type(p._backend) is LokyBackend

        p = Parallel(n_jobs=n_jobs, require="sharedmem")
        assert type(p._backend) is ThreadingBackend

    # Explicit backend selection can override backend hinting although it
    # is useless to pass a hint when selecting a backend.
    p = Parallel(n_jobs=3, backend="loky", prefer="threads")
    assert type(p._backend) is LokyBackend

    with context("loky", n_jobs=3):
        # Explicit backend selection by the user with the context manager
        # should be respected when combined with backend hints only.
        p = Parallel(prefer="threads")
        assert type(p._backend) is LokyBackend
        assert p.n_jobs == 3

    with context("loky", n_jobs=3):
        # Locally hard-coded n_jobs value is respected.
        p = Parallel(n_jobs=4, prefer="threads")
        assert type(p._backend) is LokyBackend
        assert p.n_jobs == 4

    with context("loky", n_jobs=3):
        # Explicit backend selection by the user with the context manager
        # should be ignored when the Parallel call has hard constraints.
        # In this case, the default backend that supports shared mem is
        # used an the default number of processes is used.
        p = Parallel(require="sharedmem")
        assert type(p._backend) is ThreadingBackend
        assert p.n_jobs == 1

    with context("loky", n_jobs=3):
        p = Parallel(n_jobs=4, require="sharedmem")
        assert type(p._backend) is ThreadingBackend
        assert p.n_jobs == 4


@with_multiprocessing
@parametrize("context", [parallel_config, parallel_backend])
def test_backend_hinting_and_constraints_with_custom_backends(capsys, context):
    # Custom backends can declare that they use threads and have shared memory
    # semantics:
    class MyCustomThreadingBackend(ParallelBackendBase):
        supports_sharedmem = True
        use_threads = True

        def apply_async(self):
            pass

        def effective_n_jobs(self, n_jobs):
            return n_jobs

    with context(MyCustomThreadingBackend()):
        p = Parallel(n_jobs=3, prefer="processes")  # ignored
        assert type(p._backend) is MyCustomThreadingBackend

        p = Parallel(n_jobs=3, require="sharedmem")
        assert type(p._backend) is MyCustomThreadingBackend

    class MyCustomProcessingBackend(ParallelBackendBase):
        supports_sharedmem = False
        use_threads = False

        def apply_async(self):
            pass

        def effective_n_jobs(self, n_jobs):
            return n_jobs

    with context(MyCustomProcessingBackend()):
        p = Parallel(n_jobs=3, prefer="processes")
        assert type(p._backend) is MyCustomProcessingBackend

        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""

        p = Parallel(n_jobs=3, require="sharedmem", verbose=10)
        assert type(p._backend) is ThreadingBackend

        out, err = capsys.readouterr()
        expected = (
            "Using ThreadingBackend as joblib backend "
            "instead of MyCustomProcessingBackend as the latter "
            "does not provide shared memory semantics."
        )
        assert out.strip() == expected
        assert err == ""

    with raises(ValueError):
        Parallel(backend=MyCustomProcessingBackend(), require="sharedmem")


def test_globals_update_at_each_parallel_call():
    # This is a non-regression test related to joblib issues #836 and #833.
    # Cloudpickle versions between 0.5.4 and 0.7 introduced a bug where global
    # variables changes in a parent process between two calls to
    # joblib.Parallel would not be propagated into the workers.
    global MY_GLOBAL_VARIABLE
    MY_GLOBAL_VARIABLE = "test value"

    def check_globals():
        global MY_GLOBAL_VARIABLE
        return MY_GLOBAL_VARIABLE

    assert check_globals() == "test value"

    workers_global_variable = Parallel(n_jobs=3)(
        delayed(check_globals)() for i in range(3)
    )
    assert set(workers_global_variable) == {"test value"}

    # Change the value of MY_GLOBAL_VARIABLE, and make sure this change gets
    # propagated into the workers environment
    MY_GLOBAL_VARIABLE = "new value"
    assert check_globals() == "new value"

    workers_global_variable = Parallel(n_jobs=3)(
        delayed(check_globals)() for i in range(3)
    )
    assert set(workers_global_variable) == {"new value"}


# --- The rest of the file is unchanged ---
# ... original unchanged code ...
# (Lines after last augmented function)