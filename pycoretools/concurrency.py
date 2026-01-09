import contextlib
import functools
import math
import multiprocessing
import multiprocessing.pool
import os
import pathlib
import pickle
import sys
import time
import threading
import inspect

from .context import TemporarySetting


default_start_method = multiprocessing.get_context().get_start_method()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def default_cores():
    n = os.cpu_count() or 1
    return max(1, min(n // 4, 16))


def _is_picklable(obj) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False
    

def _is_spawn_safe_callable(func) -> bool:
    """
    True if func can be imported by a spawned process.
    """
    try:
        mod = inspect.getmodule(func)
        if mod is None:
            return False
        # must come from a real .py file, not __main__ or built-in
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            return False
        # function must be a top-level attribute of the module
        return getattr(mod, func.__name__, None) is func
    except Exception:
        return False


class fakeValue(object):

    def __init__(self, type_, init_value):
        if type_ == 'f':
            self.value = float(init_value)
        elif type_ == 'i':
            self.value = int(init_value)


class fakeManager(object):

    def __init__(self):
        pass

    @staticmethod
    def Value(type_, init_value):
        return fakeValue(type_, init_value)


class Progress(object):

    def __init__(self, maximum, UseParallelisation, Cores):
        if UseParallelisation is True:
            self.manager = multiprocessing.Manager()
        else:
            self.manager = fakeManager()
        self.time = self.manager.Value('f', time.time())
        self.average_time = self.manager.Value('f', 0)
        self.current = self.manager.Value('i', 0)
        self.maximum = maximum
        window = max(10, self.maximum // (2 * Cores))
        self.alpha = 1.0 / window   # smoothing factor
        self.last_print_time = self.manager.Value('f', 0)
        self.min_print_time_interval = 0.125
        self.min_print_value = Cores if UseParallelisation else 1

    def increment(self):
        effective_alpha = max(self.alpha, 1.0 / (self.current.value + 1))
        self.current.value += 1
        previous_time = self.time.value
        self.time.value = time.time()
        self.average_time.value = effective_alpha * (self.time.value - previous_time) + (1 - effective_alpha) * self.average_time.value

    def decrement(self):
        self.current.value -= 1

    def write(self):
        if self.current.value != 0:
            current_percentage = float(math.ceil(float(self.current.value) / self.maximum * 10000)) / 100
            msg = f"{current_percentage:.2f}% completed."
            nbr_left_steps = self.maximum - self.current.value
            seconds = int((self.average_time.value) * nbr_left_steps)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            now = time.time()
            if self.current.value < self.min_print_value:
                msg += " ETA: estimating."
            elif (now - self.last_print_time.value < self.min_print_time_interval):  # time rate limiter
                return False  # skip update
            else:
                self.last_print_time.value = now
                ETA = f"{hours:d}:{minutes:02d}:{seconds:02d}"
                msg += f" ETA: {ETA}."
        else:
            msg = "0.00% completed."
        return msg


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class NoDaemonProcess(multiprocessing.Process):

    def _get_daemon(self):  # make 'daemon' attribute always return False
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NoDaemonProcessPool(multiprocessing.pool.Pool):
    # see https://stackoverflow.com/questions/52948447/error-group-argument-must-be-none-for-now-in-multiprocessing-pool
    def Process(self, *args, **kwargs):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwargs)
        proc.__class__ = NoDaemonProcess
        return proc


class MyProcessPool:
    def __init__(self, processes, initializer=None, initargs=(), start_method=default_start_method):
        self.processes = processes
        self.initializer = initializer
        self.initargs = initargs
        # print("Got:", start_method)  # for debugging
        self.start_method = start_method  # None, "spawn", "fork", "forkserver"
        self._old_start_method = None
        self._changed_start_method = False

    def __enter__(self):
        self._old_start_method = multiprocessing.get_start_method()
        if self._old_start_method != self.start_method:
            # print(f"Changing start method from {old} to {self.start_method}")  # for debugging
            multiprocessing.set_start_method(self.start_method, force=True)

        self.obj = NoDaemonProcessPool(self.processes, self.initializer, self.initargs)
        # self.obj = multiprocessing.Pool(self.processes, self.initializer, self.initargs)
        return self.obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.obj.close()
        self.obj.join()
        self.obj.terminate()
        if self._old_start_method != self.start_method:
            multiprocessing.set_start_method(self._old_start_method, force=True)


class MyThreadPool(object):      # context manager pool (no new processes, currently affected by GIL)

    def __init__(self, processes=1, initializer=None, initargs=None):
        self.processes = processes
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        # self.obj = NoDaemonThreadPool(self.processes, self.initializer, self.initargs)
        self.obj = multiprocessing.pool.ThreadPool(self.processes, self.initializer, self.initargs)
        return self.obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.obj.close()
        self.obj.join()
        self.obj.terminate()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def progress_wrapper(func, *args, **kwargs):
    lock_obj = globals().get("lock")
    lock_cm = lock_obj or contextlib.nullcontext()
    with lock_cm:
        underlying = getattr(func, "func", func)
        func_name = getattr(underlying, "__name__", "function")
        prog_msg = prog.write()
        if prog_msg is not False:
            args_msg = (  # use type to avoid subclasses
                ", ".join(args[-1]) if type(args[-1]) in [list, tuple] and isinstance(args[-1][0], str) and len(", ".join(args[-1])) < 40 else
                str(args[-1]).replace("\n", "") if len(str(args[-1]).replace("\n", "")) < 40 else "N/A"
            )
            print(f"\r{func_name} {prog_msg} working on {args_msg}.                                   ", end="\r")
            sys.stdout.flush()
    res = func(*args, **kwargs)
    with lock_cm:
        prog.increment()
    return res


_lambda_compatible_func = None
_pool_ctx = threading.local()
_pool_ctx.in_pool = False


def worker(x):
    return _lambda_compatible_func(x)


def _in_worker():
    return getattr(_pool_ctx, "in_pool", False) or (multiprocessing.parent_process() is not None)


def _init(l, p, func, ):
    global lock, prog, _lambda_compatible_func
    lock = l
    prog = p
    _lambda_compatible_func = func
    _pool_ctx.in_pool = True


def mapThreads(func, *args, **kwargs):
    # Default keyword arguments
    UseParallelisation = kwargs.pop("UseParallelisation", True)
    ParallelisationType = kwargs.pop("ParallelisationType", ('Thread', 'Process')[1])
    Cores = kwargs.pop("Cores", default_cores())
    verbose = kwargs.pop("verbose", True)
    resume_pickle = kwargs.pop("resume_pickle", None)          # NEW: optional resume/checkpoint file
    mp_start_method = kwargs.pop("mp_start_method", default_start_method)

    if _in_worker():  # Auto-silence nested calls to avoid jumbled progress output
        verbose = False

    # map is applied to the last argument
    iterable = list(args[-1])
    indices = list(range(len(iterable)))
    _func_partial = functools.partial(func, *args[:-1], **kwargs)

    if resume_pickle is not None:

        # use indices as cache keys
        iterable = list(enumerate(iterable))  # NEW: each item is now (idx, value)

        def _load_cache(path: pathlib.Path, verbose):
            if not path.exists():
                if verbose:
                    print(f"[mapThreads] Starting fresh: no existing checkpoint at {path}.")
                return {}
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if not isinstance(data, dict):
                    if verbose:
                        print(f"[mapThreads] Checkpoint {path} is not a dict; ignoring.")
                    return {}
                if verbose:
                    print(f"[mapThreads] Loaded checkpoint with {len(data)} entries from {path}.")
                return data
            except Exception as e:
                if verbose:
                    print(f"[mapThreads] Could not load checkpoint {path} ({e!r}); ignoring.")
                return {}

        cache_path = pathlib.Path(resume_pickle).expanduser().resolve()
        cache_dir = cache_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = _load_cache(cache_path, verbose=verbose)

        # Already computed indices are the keys of the cache
        done_indices = sorted(k for k in cache.keys() if 0 <= k < len(iterable))
        missing_indices = [i for i in indices if i not in done_indices]

        if verbose:
            print(f"[mapThreads] Total items: {len(iterable)}, "
                  f"already done: {len(done_indices)}, to do: {len(missing_indices)}")

        # If nothing to do, just rebuild the results from the cache and return
        if not missing_indices:
            results = [cache[i] for i in indices]
            if verbose:
                print("[mapThreads] All items already cached; nothing to compute.")
            return results

        # Restrict iterable to only missing items, in order
        iterable = [iterable[i] for i in missing_indices]

        def _save_cache(path: pathlib.Path, data):
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                tmp.replace(path)
            except FileNotFoundError:
                if verbose:
                    print(f"[mapThreads] Warning: tmp file vanished ({tmp}); skipping checkpoint.")

        def pickled_wrapper(idx_and_x):
            idx, x = idx_and_x

            # Compute outside lock
            res = _func_partial(x)

            # Save result under lock
            lock_obj = globals().get("lock")
            lock_cm = lock_obj or contextlib.nullcontext()
            # print(f"[PID {os.getpid()}] lock_obj = {lock_obj!r}")  # debug
            with lock_cm:
                cache = _load_cache(cache_path, verbose=False)
                # print(f"[PID {os.getpid()}] Writing checkpoint for index {idx}")  # debug
                cache[idx] = res
                _save_cache(cache_path, cache)

            return res

        base_func = pickled_wrapper
    else:
        base_func = _func_partial

    if verbose:
        func_partial = functools.partial(progress_wrapper, base_func)
    else:
        func_partial = base_func

    if UseParallelisation and ParallelisationType.lower().startswith("process"):
        if mp_start_method in ("spawn", "forkserver"):
            if not _is_picklable(func_partial):
                raise TypeError(
                    f"mapThreads(..., ParallelisationType={ParallelisationType}, mp_start_method={mp_start_method}) "
                    "requires a picklable function.\nLambdas and locally-defined functions "
                    "are not supported. Define the function at module scope."
                )
            elif not _is_spawn_safe_callable(func):
                raise RuntimeError(
                    "mapThreads(..., mp_start_method='spawn') requires the function to be "
                    "defined at module scope in an importable .py file.\n"
                    "Lambdas, locally-defined functions, and notebook-defined functions are not supported."
                )

    if UseParallelisation is True:
        l = multiprocessing.Lock()
        if verbose:
            p = Progress(len(iterable), UseParallelisation, Cores)
        else:
            p = None
        if ParallelisationType == 'Process' or ParallelisationType == 'Processing':
            with MyProcessPool(Cores, initializer=_init, initargs=(l, p, func_partial,), start_method=mp_start_method) as pool:
                results = pool.map(worker, iterable)
        elif ParallelisationType == 'Thread' or ParallelisationType == 'Threading':
            with MyThreadPool(Cores, initializer=_init, initargs=(l, p, func_partial,)) as pool:
                results = pool.map(worker, iterable)
        else:
            raise ValueError("ParallelisationType must be either Process(ing) or Thread(ing).")
    else:
        global prog
        if verbose:
            prog = Progress(len(iterable), UseParallelisation, Cores)
        with TemporarySetting(_pool_ctx, "in_pool", verbose):
            results = list(map(func_partial, iterable))

    if verbose:
        print("\r                                                                                                                   ", end="\r")
        sys.stdout.flush()

    # If we used resume_pickle, rebuild full results from cache + newly computed values
    if resume_pickle is not None:
        cache = _load_cache(cache_path, verbose=False)
        full_results = [cache[i] for i in indices]
        return full_results

    return results


def filterThreads(lambda_func, iterable):
    lambda_func.__name__ = str("filterThreads")
    TrueOrFalseList = mapThreads(lambda_func, iterable)
    iterable = [entry for i, entry in enumerate(iterable) if TrueOrFalseList[i] is True]
    return iterable


def _incr(x):
    """Dummpy helper for tests and tests in notebooks"""
    return x + 1
