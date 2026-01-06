import pytest

from pycoretools.iterables import crease, flatten


def test_crease_trivial():
    template = [None, None, None]
    iterable = [1, 2, 3]
    assert crease(iterable, template, depth=0) == [1, 2, 3]


@pytest.mark.parametrize(
    "template, depth, data",   # data is also expected result
    [
        # uniform depth
        ([[None, None], [None, None, None]], 1, [[1, 2], [3, 4, 5]]),
        # rugged depth
        ([[None], [[None], [None, None, None]], [None]], 2, [[1], [[2], [3, 4, 5]], [6]]),
        ([[[None], [None, None, None]], [None], [None]], 2, [[[1], [2, 3, 4]], [5], [6]]),
        # rugged depth, template gets chopped
        ([[None], [[None], [None, [None, None], None]], [None]], 2, [[1], [[2], [3, 4, 5]], [6]]),
        # rugged shape supported (depth=1; means “shape only at one level”)
        ([[None], [None, None, None], []], 1, [[1], [2, 3, 4], []]),
    ],
)
def test_crease_roundtrip_known_cases(template, depth, data):
    flat = flatten(data)
    rebuilt = crease(flat, template, depth=depth)

    assert rebuilt == data
    assert flatten(rebuilt) == flat


def test_crease_raises_on_length_mismatch():
    template = [[None, None], [None]]  # expects 3 elements at depth=1
    with pytest.raises(AssertionError):
        crease([1, 2], template, depth=1)


def test_crease_negative_depth_raises():
    with pytest.raises(ValueError):
        crease([1, 2, 3], [None, None, None], depth=-1)
