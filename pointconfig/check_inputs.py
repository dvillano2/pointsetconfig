"""
Functions to check viability of various arguments
"""

from pointconfig.config_types import DirectionType
from pointconfig.config_types import PointType


def check_prime_dim(prime: int, dimension: int) -> None:
    """
    Basic checks for bad prime and dimension inputs
    """
    if not isinstance(prime, int) or not isinstance(dimension, int):
        raise TypeError("Prime and dimension must both be integers.")
    if prime < 1:
        raise ValueError("Prime must be a positive integer.")
    if dimension < 0:
        raise ValueError("Dimension must be a nonnegative integer.")


def check_prime_dim_point_dir(
    prime: int,
    dimension: int,
    point: PointType,
    direction: DirectionType,
) -> DirectionType:
    """
    Basic checks form bad prime, dimension, point, and direction inputs
    """
    check_prime_dim(prime, dimension)
    for point_or_dir in (point, direction):
        if not (
            isinstance(point_or_dir, tuple)
            and all(isinstance(coord, int) for coord in point_or_dir)
        ):
            raise TypeError(
                "Point and direction must both be tuples of integers"
            )
    if len(direction) != dimension:
        raise ValueError(
            "Direction must have dimension agreeing with dimension parameter"
        )
    if len(point) != dimension:
        raise ValueError(
            "Point must have dimension agreeing with dimension parameter"
        )
    mod_p_direction = tuple(coordinate % prime for coordinate in direction)
    if mod_p_direction and all(
        coordinate == 0 for coordinate in mod_p_direction
    ):
        raise ValueError("direction must be nonzero mod p")
    return mod_p_direction
