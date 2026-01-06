import itertools

from copy import deepcopy


def _is_numpy_ndarray(x) -> bool:
    t = type(x)
    return t.__module__ == "numpy" and t.__name__ == "ndarray"


def _is_sympy_mutable_dense_matrix(x) -> bool:
    t = type(x)
    return t.__module__ == "sympy.matrices.dense" and t.__name__ == "MutableDenseMatrix"


def flatten(temp_list, recursion_level: int = 0, treat_list_subclasses_as_list: bool = True,
            treat_tuples_as_lists: bool = False, max_recursion=None):
    
    flat_list = []

    for entry in temp_list:
        if max_recursion is not None and recursion_level >= max_recursion:
            flat_list.append(entry)
            continue

        t = type(entry)

        should_recurse = (
            t is list
            or (
                treat_list_subclasses_as_list
                and (
                    isinstance(entry, list)
                    or _is_numpy_ndarray(entry)
                    or _is_sympy_mutable_dense_matrix(entry)
                )
            )
            or (treat_tuples_as_lists and t is tuple)
        )

        if should_recurse:
            flat_list.extend(
                flatten(entry, recursion_level + 1, treat_list_subclasses_as_list, treat_tuples_as_lists, max_recursion, )
            )
        else:
            flat_list.append(entry)

    return flat_list


def crease(iterable, template, depth, called_recursively=False, verbose=False):
    """Inverse function to flatten. Requires a template to define the shape. Rugged shape is supported."""
    if verbose:
        print(f"crease called at depth {depth}")
    if not called_recursively:
        iterable = flatten(iterable)           # make sure it's flat
        creased_iterable = deepcopy(template)  # make a copy of template to return result, without deleting the template
    else:
        creased_iterable = template
    if verbose:
        print(f"len(iterable): {len(iterable)}, len(flatten(template, max_recursion={depth})): {len(flatten(template, max_recursion=depth))}")
    assert len(iterable) == len(flatten(template, max_recursion=depth))
    if depth == 0:
        for i, _ in enumerate(creased_iterable):
            creased_iterable[i] = iterable[i]
        assert flatten(creased_iterable) == iterable
        return creased_iterable
    elif depth > 0:
        for i, _ in enumerate(creased_iterable):
            if verbose:
                print(f"slice: {len(flatten(creased_iterable[:i], max_recursion=1))}:{len(flatten(creased_iterable[:i + 1], max_recursion=1))}")
            ith_iterable = iterable[len(flatten(creased_iterable[:i], max_recursion=1)):len(flatten(creased_iterable[:i + 1], max_recursion=1))]
            creased_iterable[i] = crease(ith_iterable, creased_iterable[i], depth=depth - 1, called_recursively=True, verbose=verbose)
        assert flatten(creased_iterable) == iterable
        return creased_iterable
    else:
        raise ValueError("crease called with negative depth.")


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def all_non_empty_subsets(iterable):
    return itertools.chain(*map(lambda x: itertools.combinations(iterable, x), range(1, len(iterable) + 1)))
