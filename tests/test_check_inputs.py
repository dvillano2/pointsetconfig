"""
Test bad input handling
"""

import pytest

from pointconfig.check_inputs import check_prime_dim
from pointconfig.check_inputs import check_prime_dim_point_dir


@pytest.mark.parametrize("prime, dim", [(-1, 7), (7, -4), (-7, -4)])
def test_check_prime_dim_negative(prime, dim) -> None:
    """
    Tests that correct error rasied when given negative dimension.
    """
    with pytest.raises(ValueError):
        check_prime_dim(prime, dim)


@pytest.mark.parametrize(
    "prime, dim",
    [
        (1.1, 0),
        (0, 3.8),
        (-1.4, 0),
        (0, 4.3),
        (1.1, -7),
        (1.1, -7.7),
        (-7, 4.8),
        (-7.9, 4.8),
        (1.1, 7),
        (7, 4.8),
        (7.2, 4.8),
    ],
)
def test_check_prime_dim_float(prime, dim) -> None:
    """
    Tests that correct error is raise when given a float.
    """
    with pytest.raises(TypeError):
        check_prime_dim(prime, dim)


@pytest.mark.parametrize("prime, dim", [(-3, 5), (0, 5)])
def test_check_prime_dim_nonpositive_prime(prime, dim) -> None:
    """
    Tests that correct error rasied when given nonpositive prime.
    """
    with pytest.raises(ValueError):
        check_prime_dim(prime, dim)


@pytest.mark.parametrize(
    "prime, dim, point, direction",
    [
        (7, 3, 3, (5, 6)),
        (7, 3, (5, 6), 4),
        (7, 3, (2, 3, 5), (3, 5.5, 6)),
        (7, 3, (5, 6.2, 8), (1, 4, 5)),
    ],
)
def test_check_prime_dim_point_dir_int_tuples(prime, dim, point, direction):
    """
    Tests that correct error raised when
    given incorrect types for point and direction.
    """
    with pytest.raises(TypeError):
        check_prime_dim_point_dir(prime, dim, point, direction)


@pytest.mark.parametrize(
    "prime, dim, point, direction",
    [
        (7, 3, (3,), (3, 5, 6)),
        (7, 3, (5, 6, 6), (5, 6, 1, 2)),
    ],
)
def test_check_prime_dim_point_dir_dim_length(prime, dim, point, direction):
    """
    Tests that correct error raised when
    point or direction don't have the lenth of dimension.
    """
    with pytest.raises(ValueError):
        check_prime_dim_point_dir(prime, dim, point, direction)


@pytest.mark.parametrize(
    "prime, dim, point, direction",
    [
        (5, 3, (3, 5, 6), (0, 5, 100)),
        (7, 4, (0, 0, 0, 1), (0, -14, 49, 7)),
    ],
)
def test_check_prime_dim_point_dir_zero(prime, dim, point, direction):
    """
    Tests that correct error raised when
    when direction is zero mod prime.
    """
    with pytest.raises(ValueError):
        check_prime_dim_point_dir(prime, dim, point, direction)


@pytest.mark.parametrize(
    "prime, dim, point, direction, mod_p_direction",
    [
        (5, 3, (3, 5, 6), (1, 5, 104), (1, 0, 4)),
        (7, 4, (0, 0, 0, 1), (4, -15, 49, 17), (4, 6, 0, 3)),
    ],
)
def test_check_prime_dim_point_mod_p_reduction(
    prime, dim, point, direction, mod_p_direction
):
    """
    Tests that direction reduction mod prime is correct.
    """
    assert (
        check_prime_dim_point_dir(prime, dim, point, direction)
        == mod_p_direction
    )
