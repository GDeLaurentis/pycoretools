import numpy
import pytest
import random
import functools

from multiprocessing import Pool
from pycoretools.concurrency import mapThreads, MyProcessPool, _init, _in_worker, worker


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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
