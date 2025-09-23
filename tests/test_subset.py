"""
Tests for the Subset class
"""

# pylint: disable=protected-access
import itertools
import pytest
from pointconfig.subset import Subset
from pointconfig.config_types import PointType


def test_lookup_updating() -> None:
    """
    Tests the behavior of the shared lookup table.
    """
    subset_0 = Subset(5, 3)
    assert len(Subset._LOOKUP) == 1
    subset_1 = Subset(5, 3)
    assert len(Subset._LOOKUP) == 1
    assert subset_0._LOOKUP is subset_1._LOOKUP
    assert subset_0.lookup_update_required
    assert not subset_1.lookup_update_required
    subset_2 = Subset(7, 3)
    assert subset_2.lookup_update_required
    assert subset_2._LOOKUP is subset_1._LOOKUP
    assert len(Subset._LOOKUP) == 2


vector_spaces = [(3, 2), (3, 4), (5, 2), (7, 3)]


@pytest.mark.parametrize("prime, dimension", vector_spaces)
def test_initialization(prime, dimension) -> None:
    """
    Tests that everything gets initialzed correctly
    """
    subset_0 = Subset(prime, dimension)
    _initial_assertions(prime, dimension, subset_0)


points = [(0, 1), (0, 1, 1, 0), (4, 2), (1, 2, 6)]
vector_spaces_with_points = [
    (prime, dimension, point)
    for (prime, dimension), point in zip(vector_spaces, points)
]


@pytest.mark.parametrize("prime, dimension, point", vector_spaces_with_points)
def test_add_remove_single_point(prime, dimension, point) -> None:
    subset_0 = Subset(prime, dimension)
    subset_0.add_point(point)
    _one_point_assertions(prime, dimension, point, subset_0)
    subset_0.remove_point(point)
    _initial_assertions(prime, dimension, subset_0)


def _add_zeros(twod_point: PointType, dimension: int) -> PointType:
    return twod_point + tuple(0 for _ in range(dimension - 2))


@pytest.mark.parametrize("prime, dimension", vector_spaces)
def test_geometric_structure(prime, dimension) -> None:
    """
    More complicated than other tests
    Needs a serious refactor
    """
    subset_0 = Subset(prime, dimension)
    zero_point = _add_zeros((0, 0), dimension)
    subset_0.add_point(zero_point)
    ones_point = _add_zeros((1, 1), dimension)
    subset_0.add_point(ones_point)
    assert subset_0.size == 2
    assert ones_point in subset_0.point_pairs_per_direction
    assert zero_point in subset_0.directions_per_point
    assert ones_point in subset_0.directions_per_point
    for direction in subset_0.line_incidence:
        if direction != ones_point:
            assert all(
                line_points in {0, 1}
                for line_points in subset_0.line_incidence[direction].values()
            )
            assert sum(subset_0.line_incidence[direction].values()) == 2
        else:
            assert all(
                line_points in {0, 2}
                for line_points in subset_0.line_incidence[direction].values()
            )
            assert sum(subset_0.line_incidence[direction].values()) == 2
    for normal_direction in subset_0.plane_incidence:
        line_containment = sum(normal_direction[:2]) % prime
        if line_containment != 0:
            assert all(
                plane_points in {0, 1}
                for plane_points in subset_0.plane_incidence[
                    normal_direction
                ].values()
            )
        else:
            assert all(
                plane_points in {0, 2}
                for plane_points in subset_0.plane_incidence[
                    normal_direction
                ].values()
            )
        assert sum(subset_0.plane_incidence[normal_direction].values()) == 2
    twos_point = _add_zeros((2, 2), dimension)
    subset_0.add_point(twos_point)
    assert subset_0.size == 3
    assert ones_point in subset_0.point_pairs_per_direction
    assert len(subset_0.point_pairs_per_direction) == 1
    assert len(subset_0.point_pairs_per_direction[ones_point]) == 3
    # this is a rote repeat of above with 3 instead of 2, should refactor
    for direction in subset_0.line_incidence:
        if direction != ones_point:
            assert all(
                line_points in {0, 1}
                for line_points in subset_0.line_incidence[direction].values()
            )
            assert sum(subset_0.line_incidence[direction].values()) == 3
        else:
            assert all(
                line_points in {0, 3}
                for line_points in subset_0.line_incidence[direction].values()
            )
            assert sum(subset_0.line_incidence[direction].values()) == 3
    for normal_direction in subset_0.plane_incidence:
        line_containment = sum(normal_direction[:2]) % prime
        if line_containment != 0:
            assert all(
                plane_points in {0, 1}
                for plane_points in subset_0.plane_incidence[
                    normal_direction
                ].values()
            )
        else:
            assert all(
                plane_points in {0, 3}
                for plane_points in subset_0.plane_incidence[
                    normal_direction
                ].values()
            )
        assert sum(subset_0.plane_incidence[normal_direction].values()) == 3
    # get rid of line point, add a third point in general position
    subset_0.remove_point(twos_point)
    one_zero_point = _add_zeros((1, 0), dimension)
    subset_0.add_point(one_zero_point)
    assert len(subset_0.point_pairs_per_direction) == 3
    for set_of_pairs in subset_0.point_pairs_per_direction.values():
        assert len(set_of_pairs) == 1
    for point in subset_0.points:
        assert point in subset_0.directions_per_point
        assert len(subset_0.directions_per_point[point]) == 2

    special_directions = (
        ones_point,
        one_zero_point,
        _add_zeros((0, 1), dimension),
    )
    # this is a rote repeat again, refactor needed
    for direction in subset_0.line_incidence:
        if direction not in special_directions:
            assert all(
                line_points in {0, 1}
                for line_points in subset_0.line_incidence[direction].values()
            )
            assert sum(subset_0.line_incidence[direction].values()) == 3
        else:
            assert all(
                line_points in {0, 1, 2}
                for line_points in subset_0.line_incidence[direction].values()
            )
            assert sum(subset_0.line_incidence[direction].values()) == 3
    if dimension > 2:
        for normal_direction in subset_0.plane_incidence:
            triangle_containment = (
                normal_direction[0] == 0 and normal_direction[1] == 0
            )
            line_containment = (
                normal_direction[0] == 0
                or normal_direction[1] == 0
                or sum(normal_direction[:2]) % prime == 0
            )

            if triangle_containment:
                assert all(
                    plane_points in {0, 3}
                    for plane_points in subset_0.plane_incidence[
                        normal_direction
                    ].values()
                )
            elif line_containment:
                assert all(
                    plane_points in {0, 1, 2}
                    for plane_points in subset_0.plane_incidence[
                        normal_direction
                    ].values()
                )
            else:
                assert all(
                    plane_points in {0, 1}
                    for plane_points in subset_0.plane_incidence[
                        normal_direction
                    ].values()
                )
            assert (
                sum(subset_0.plane_incidence[normal_direction].values()) == 3
            )


def _initial_assertions(prime, dimension, initialized_subset) -> None:
    assert initialized_subset.space.prime == prime
    assert initialized_subset.space.dimension == dimension
    assert not initialized_subset.points
    assert not initialized_subset.point_pairs_per_direction
    assert not initialized_subset.directions_per_point

    total_directions = sum(prime**power for power in range(dimension))
    # plane incidence
    assert len(initialized_subset.plane_incidence) == total_directions
    for intercept in range(prime):
        for normal_direction in initialized_subset.plane_incidence:
            assert len(normal_direction) == dimension
            assert (
                intercept
                in initialized_subset.plane_incidence[normal_direction]
            )
            assert (
                initialized_subset.plane_incidence[normal_direction][intercept]
                == 0
            )
    # line incidence
    assert len(initialized_subset.line_incidence) == total_directions
    for line_intercept in itertools.product(
        range(prime), repeat=dimension - 1
    ):
        for direction in initialized_subset.line_incidence:
            assert len(direction) == dimension
            assert (
                line_intercept in initialized_subset.line_incidence[direction]
            )
            assert (
                initialized_subset.line_incidence[direction][line_intercept]
                == 0
            )


def _one_point_assertions(prime, dimension, point, one_point_subset) -> None:
    assert one_point_subset.size == 1
    assert not one_point_subset.directions_per_point
    assert not one_point_subset.point_pairs_per_direction
    for normal_direction in one_point_subset.plane_incidence:
        true_intercept = sum(
            (point_coord * normal_coord)
            for point_coord, normal_coord in zip(point, normal_direction)
        )
        true_intercept = true_intercept % prime
        for intercept in range(prime):
            plane_points = one_point_subset.plane_incidence[normal_direction][
                intercept
            ]
            if intercept == true_intercept:
                assert plane_points == 1
            else:
                assert plane_points == 0
    with pytest.raises(KeyError):
        one_point_subset.remove_point(tuple(1 for _ in range(dimension)))
