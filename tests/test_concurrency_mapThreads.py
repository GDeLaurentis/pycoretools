import pytest
import multiprocessing
import multiprocessing.pool as mp_pool
import inspect
import functools
import time

from multiprocessing import Pool
from pycoretools.concurrency import mapThreads, MyProcessPool, _init, _in_worker, worker, _incr, default_start_method


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_multiprocessing_pool_repopulate_signature_stable():
    """
    Guard against CPython changes in Pool internals.

    pycoretools overrides Pool._repopulate_pool_static to disable daemon workers.
    If the stdlib signature changes, the override must be updated.
    """
    expected = (
        "ctx",
        "Process",
        "processes",
        "pool",
        "inqueue",
        "outqueue",
        "initializer",
        "initargs",
        "maxtasksperchild",
        "wrap_exception",
    )

    sig = inspect.signature(mp_pool.Pool._repopulate_pool_static)
    assert tuple(sig.parameters) == expected, (
        "multiprocessing.pool.Pool._repopulate_pool_static signature changed; "
        "update NoDaemonProcessPool override."
    )


def test_mapThreads_simple():
    def shift(x):
        return x + 1
    assert mapThreads(lambda x: x + 1, range(100), Cores=4) == mapThreads(shift, range(100), Cores=4) == list(range(1, 101))


def test_mapThreads_multiple_arguments():
    def shift(x, y):
        return x + y
    assert mapThreads(shift, 5, range(100), Cores=4) == list(range(5, 105))


def test_mapThreads_nodaemon():
    def shift(xs, y):
        return sum(mapThreads(lambda x: x + 1, xs, Cores=4)) + y
    assert mapThreads(shift, range(100), range(100), Cores=4) == list(range(5050, 5150))


def f(x):
    return sum(mapThreads(lambda x: x + 1, range(x), UseParallelisation=False))


def g(x):
    return sum(mapThreads(lambda x: x + 1, range(x), UseParallelisation=True))


def test_daemonic_parallelisation_off():
    with Pool() as pool:
        res = pool.map(f, range(10))
    assert sum(res) == 165


def test_daemonic_parallelisation_on():
    with pytest.raises(AssertionError):
        with Pool() as pool:
            pool.map(g, range(10))


def test_in_worker_is_aware_of_being_in_pool():
    assert _in_worker() is False  # main process should not be marked as worker

    def inner_check(_):
        return _in_worker()

    # minimal init args (no progress, no lock) just to trigger _init
    l, p = None, None
    func_partial = inner_check

    with MyProcessPool(2, initializer=_init, initargs=(l, p, func_partial)) as pool:
        results = pool.map(worker, range(2))

    assert all(results)


def test_mapThreads_threading():
    assert mapThreads(_incr, range(100), Cores=4, ParallelisationType='Thread', verbose=False) == list(range(1, 101))


def test_mapThreads_spawn():
    assert mapThreads(_incr, range(100), Cores=4, mp_start_method='spawn', verbose=False) == list(range(1, 101))


def test_global_context_unaffected_by_start_method():
    assert multiprocessing.get_context().get_start_method() == default_start_method
    assert mapThreads(_incr, range(100), Cores=4, mp_start_method='spawn', verbose=False) == list(range(1, 101))
    assert multiprocessing.get_context().get_start_method() == default_start_method
    assert mapThreads(_incr, range(100), Cores=4, mp_start_method='fork', verbose=False) == list(range(1, 101))
    assert multiprocessing.get_context().get_start_method() == default_start_method


def _inner_concurrency_task(value):
    return value * value


def _outer_concurrency_task(value, inner_parallelisation_type):
    time.sleep(0.15)
    inner_results = mapThreads(
        _inner_concurrency_task, [1, 2, 3], Cores=2, ParallelisationType=inner_parallelisation_type, UseParallelisation=True, verbose=True,
    )
    return value + sum(inner_results)


def _test_nested_concurrency(outer_parallelisation_type, inner_parallelisation_type):
    outer_task = functools.partial(_outer_concurrency_task, inner_parallelisation_type=inner_parallelisation_type)
    results = mapThreads(
        outer_task, [10, 20, 30, 40, 50, 60, 70, 80, 90],
        Cores=2, ParallelisationType=outer_parallelisation_type, UseParallelisation=True, verbose=True, progress_new_line=True,
    )
    assert results == [24, 34, 44, 54, 64, 74, 84, 94, 104]


def test_nesting_process_and_thread_concurrency():
    _test_nested_concurrency("Process", "Thread")


def test_nesting_thread_and_process_concurrency():
    _test_nested_concurrency("Thread", "Process")
