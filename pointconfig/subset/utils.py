"""
Helper functions for making lookup tables for incidence info
main function is found in lookup.py
get_directions is also used directly in the subset class
"""

import itertools
from collections.abc import Generator
from typing import Dict
from typing import Tuple

import pointconfig.check_inputs as checks
from pointconfig.config_types import DirectionType
from pointconfig.config_types import LookupEntryType
from pointconfig.config_types import PointType


def get_directions(
    prime: int, dimension: int
) -> Generator[DirectionType, None, None]:
    """
    Gives all the directions in the vector space of
    given dimension over the field with the given
    prime number of elements
    """
    checks.check_prime_dim(prime, dimension)
    if dimension == 0:
        return
    flat_directions = (
        lower_dim_direction + (0,)
        for lower_dim_direction in get_directions(prime, dimension - 1)
    )
    sloped_directions = (
        slope + (1,)
        for slope in itertools.product(range(prime), repeat=dimension - 1)
    )
    yield from itertools.chain(sloped_directions, flat_directions)


def get_plane_paramterizing_intercept(
    prime: int,
    dimension: int,
    point: PointType,
    normal_direction: DirectionType,
) -> int:
    """
    Given a point and a nomal direction, gives the
    'intercept' of the plane containing the point normal
    to the normal direction.
    """
    mod_p_direction = checks.check_prime_dim_point_dir(
        prime, dimension, point, normal_direction
    )
    dot_product = sum(
        point_coord * mod_p_coord
        for point_coord, mod_p_coord in zip(point, mod_p_direction)
    )
    return dot_product % prime


def get_line_paramterizing_intercept(
    prime: int,
    dimension: int,
    point: PointType,
    direction: DirectionType,
) -> PointType:
    """
    Given a line and a point, gives the intercept that
    tells you which of the parallel lines the point
    lies on. This is the dimension - 1 tuple obtained by
    finding the last nonzero (mod p) entry in the direction
    translating the point along the line until its
    zero and then deleting that coordinate
    """
    # Nonsensical inputs
    mod_p_direction = checks.check_prime_dim_point_dir(
        prime, dimension, point, direction
    )
    # Actually getting the intercept
    # Normalize the direction
    nonzero_index = -1
    if not mod_p_direction:
        return ()
    while mod_p_direction[nonzero_index] == 0:
        nonzero_index -= 1
    normalizer = pow(mod_p_direction[nonzero_index], -1, prime)
    normalized_direction = (
        (coord * normalizer) % prime for coord in mod_p_direction
    )

    # Shift the point
    mod_p_point = tuple(coord % prime for coord in point)
    key_coord = mod_p_point[nonzero_index]
    intercept = [
        (point_coord - (dir_coord * key_coord)) % prime
        for point_coord, dir_coord in zip(mod_p_point, normalized_direction)
    ]

    # Get rid of the guarateed zero coordinate
    del intercept[nonzero_index]
    return tuple(intercept)


def create_lookup_entry(
    prime: int, dimension: int, point: PointType
) -> LookupEntryType:
    """
    For a given point, goes through every direction of the space and
    tells you which of the planes it lies on normal to that direction
    and which of the lines it lies on normal to that direction
    """
    containment_info: LookupEntryType = {
        "planes": {},
        "lines": {},
    }
    directions = get_directions(prime, dimension)
    for direction in directions:
        containment_info["planes"][direction] = (
            get_plane_paramterizing_intercept(
                prime, dimension, point, direction
            )
        )
        containment_info["lines"][direction] = (
            get_line_paramterizing_intercept(
                prime, dimension, point, direction
            )
        )
    return containment_info


def create_full_lookup_table(
    prime: int, dimension: int
) -> Dict[PointType, LookupEntryType]:
    """
    Returns a full lookup table for the given prime and dimension
    """
    lookup_table: Dict[PointType, LookupEntryType] = {}
    for point in itertools.product(range(prime), repeat=dimension):
        lookup_table[point] = create_lookup_entry(prime, dimension, point)
    return lookup_table
