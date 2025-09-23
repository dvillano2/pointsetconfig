"""
Test the util functions
"""

import pytest

from pointconfig.utils import get_directions
from pointconfig.utils import get_plane_paramterizing_intercept
from pointconfig.utils import get_line_paramterizing_intercept
from pointconfig.utils import create_lookup_entry
from pointconfig.utils import create_full_lookup_table


def test_get_directions() -> None:
    """
    Tests that correct error rasied when given float dimension.
    """
    dimension_zero = get_directions(3, 0)
    dimension_one_0 = get_directions(5, 1)
    dimension_one_1 = get_directions(557, 1)
    dimension_two = get_directions(5, 2)
    dimension_three = get_directions(3, 3)
    high_dimension = get_directions(7, 5)

    dim_three_answer_2 = [
        (coord_0, coord_1, 1) for coord_0 in range(3) for coord_1 in range(3)
    ]
    dim_three_answer_1 = [(slope, 1, 0) for slope in range(3)]
    dim_three_answer_0 = [(1, 0, 0)]
    high_dim_answer_length = sum(7**power for power in range(5))

    assert not list(dimension_zero)
    assert list(dimension_one_0) == [(1,)]
    assert list(dimension_one_1) == [(1,)]
    assert list(dimension_two) == [(slope, 1) for slope in range(5)] + [(1, 0)]
    assert (
        list(dimension_three)
        == dim_three_answer_2 + dim_three_answer_1 + dim_three_answer_0
    )
    assert len(list(high_dimension)) == high_dim_answer_length


def test_get_plane_paramterizing_intercept_zero_direction() -> None:
    """
    Tests that correct plane is computed
    """
    with pytest.raises(ValueError):
        get_plane_paramterizing_intercept(5, 3, (3, 5, 1), (5, -20, 0))


geometric_test_cases = [
    (7, 0, (), (), 0, ()),
    (7, 1, (4,), (2,), 1, ()),
    (11, 1, (9,), (12,), 9, ()),
    (11, 1, (9,), (7,), 8, ()),
    (5, 2, (2, 4), (3, 1), 0, (0,)),
    (5, 3, (2, 4, 1), (1, 3, 1), 0, (1, 1)),
    (11, 6, (2, 6, 8, 3, 5, 1), (1, 5, 75, 2, 1, 5), 10, (4, 5, 4, 7, 7)),
    (11, 6, (2, 6, 8, 3, 5, 1), (1, 5, 75, 2, 1, 0), 5, (8, 3, 7, 4, 1)),
]


@pytest.mark.parametrize(
    "prime, dim, point, direction, expected_value, _", geometric_test_cases
)
def test_get_plane_paramterizing_intercept(
    prime, dim, point, direction, expected_value, _
) -> None:
    assert (
        get_plane_paramterizing_intercept(prime, dim, point, direction)
        == expected_value
    )


@pytest.mark.parametrize(
    "prime, dim, point, direction, _, expected_value", geometric_test_cases
)
def test_get_line_paramterizing_intercept(
    prime, dim, point, direction, _, expected_value
) -> None:
    assert (
        get_line_paramterizing_intercept(prime, dim, point, direction)
        == expected_value
    )


def test_create_lookup_entry() -> None:
    zero_dim = create_lookup_entry(13, 0, ())
    one_dim = create_lookup_entry(13, 1, (8,))
    two_dim = create_lookup_entry(13, 2, (8, 9))
    three_dim = create_lookup_entry(13, 3, (0, 8, 9))
    planes_lines = ["planes", "lines"]
    for example in (zero_dim, one_dim, two_dim, three_dim):
        assert list(example.keys()) == planes_lines
    for pl in planes_lines:
        assert not zero_dim[pl]
        assert len(two_dim[pl]) == 14
        assert len(three_dim[pl]) == 13**2 + 13 + 1
    assert len(one_dim["planes"]) == 1
    assert (1,) in one_dim["planes"]
    assert one_dim["lines"] == {(1,): ()}


def test_create_full_lookup_table() -> None:
    zero_dim = create_full_lookup_table(5, 0)
    one_dim = create_full_lookup_table(5, 1)
    two_dim = create_full_lookup_table(5, 2)
    three_dim = create_full_lookup_table(5, 3)

    for i, example in enumerate([zero_dim, one_dim, two_dim, three_dim]):
        assert len(example) == 5 ** i
