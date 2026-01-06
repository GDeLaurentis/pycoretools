import pytest

from pycoretools.iterables import flatten


def test_flatten_basic():
    assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]


def test_flatten_does_not_flatten_tuples_by_default():
    assert flatten([1, (2, 3), [4]]) == [1, (2, 3), 4]


def test_flatten_can_flatten_tuples_when_enabled():
    assert flatten([1, (2, (3, 4)), 5], treat_tuples_as_lists=True) == [1, 2, 3, 4, 5]


def test_flatten_respects_max_recursion():
    x = [1, [2, [3, [4]]]]
    # flatten only one nesting level
    assert flatten(x, max_recursion=1) == [1, 2, [3, [4]]]


def test_flatten_list_subclasses_flag():
    class MyList(list):
        pass

    x = [1, MyList([2, [3]]), 4]

    assert flatten(x, treat_list_subclasses_as_list=False) == [1, x[1], 4]
    assert flatten(x, treat_list_subclasses_as_list=True) == [1, 2, 3, 4]


def test_flatten_empty_cases():
    assert flatten([]) == []
    assert flatten([[], [[]]]) == []


def test_flatten_numpy_ndarray_if_numpy_installed():
    np = pytest.importorskip("numpy")
    x = [1, np.array([2, [3, 4]], dtype=object), 5]
    assert flatten(x) == [1, 2, 3, 4, 5]


def test_flatten_sympy_mutable_dense_matrix_if_sympy_installed():
    sp = pytest.importorskip("sympy")
    MD = sp.matrices.dense.MutableDenseMatrix([[1, 2], [3, 4]])
    assert flatten([MD]) == [1, 2, 3, 4]
