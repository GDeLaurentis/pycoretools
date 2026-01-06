import functools


def retry(ExceptionsToCheck, max_tries=2, silent=False):
    def deco_retry(func):
        @functools.wraps(func)
        def f_retry(*args, **kwargs):
            current_try = 0
            while current_try < max_tries:
                try:
                    return_value = func(*args, **kwargs)
                except ExceptionsToCheck as e:
                    last_exception = e
                    if silent is False:
                        print("\r{}: {}...  ".format(type(e).__name__, str(e)[:75]), end="")
                        if type(args[-1]) in [list, tuple]:
                            print(", ".join(args[-1]), end="")
                        else:
                            print("Last arg: {}".format(args[-1]), end="")
                    current_try += 1
                else:
                    if current_try != 0 and silent is False:
                        print("\r{}: {}... ".format(type(last_exception).__name__, str(last_exception)[:75]), end="")
                        if type(args[-1]) in [list, tuple]:
                            print(", ".join(args[-1]), end="")
                        else:
                            print("Last arg: {}".format(args[-1]), end="")
                        print(" ~ Corrected on try {}!                                                               ".format(current_try + 1))
                    return return_value
            if silent is False:
                print("\r{}: {}... ".format(type(last_exception).__name__, str(last_exception)[:75]), end="")
                if type(args[-1]) in [list, tuple]:
                    print(", ".join(args[-1]), end="")
                else:
                    print("Last arg: {}".format(args[-1]), end="")
                print(" ~ Uncorrected after {} tries. :(                                                           ".format(max_tries))
            return None
        return f_retry
    return deco_retry


def with_cm(cm):
    """
    A decorator that applies a context manager to the function.
    This allows a `with` statement to be used as a decorator.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with cm:
                return func(*args, **kwargs)
        return wrapper
    return decorator
