import numpy as np
from numba import njit 


@njit
def get_direction_subdimension(prime, dimension, direction_index):
    sub_dimension = 0
    threshold = ((prime**sub_dimension) - 1) // (prime - 1)
    while threshold <= direction_index:
        sub_dimension += 1
        threshold = ((prime**sub_dimension) - 1) // (prime - 1)
    return sub_dimension - 1


@njit
def get_plane_intercept_by_index(
    prime, dimension, point_index, normal_direction_index
):
    sub_dimension = get_direction_subdimension(
        prime, dimension, normal_direction_index
    )
    point_expand = point_index
    # rifle through point dimensions where direction is zero
    for _ in range(dimension - 1, sub_dimension, -1):
        point_expand //= prime
    # add where the dirction is one
    dot_product = point_expand % prime
    point_expand //= prime
    # dot product on what remains
    shift = ((prime**sub_dimension) - 1) // (prime - 1)
    direction_expand = normal_direction_index - shift
    for _ in range(sub_dimension, -1, -1):
        point_coord = point_expand % prime
        direction_coord = direction_expand % prime
        dot_product += (point_coord * direction_coord) % prime

        point_expand //= prime
        direction_expand //= prime

    return dot_product % prime


@njit
def compute_key_coordinate(prime, dimension, sub_dimension, point_index):
    key_expand = point_index
    for _ in range(dimension - 1, sub_dimension, -1):
        key_expand //= prime
    return key_expand % prime


@njit
def get_line_intercept_by_index(
    prime, dimension, point_index, direction_index
):
    sub_dimension = get_direction_subdimension(
        prime, dimension, direction_index
    )

    key_coord = compute_key_coordinate(
        prime, dimension, sub_dimension, point_index
    )

    point_expand = point_index
    direction_expand = direction_index
    intercept_index = 0
    intercept_to_index_power = 0
    for i in range(dimension - 1, -1, -1):
        if i != sub_dimension:
            point_coord = point_expand % prime
            direction_coord = direction_expand % prime
            interecept_coord = (
                point_coord - (direction_coord * key_coord)
            ) % prime
            intercept_index += interecept_coord * (
                prime ** (intercept_to_index_power)
            )
            intercept_to_index_power += 1
        point_expand //= prime
        direction_expand //= prime
    return intercept_index


def plane_lookup_factory(prime, dimension):
    @njit
    def plane_lookup():
        total_points = prime**dimension
        total_directions = ((prime**dimension) - 1) // (prime - 1)
        lookup = np.empty((total_points, total_directions), dtype=np.uint8)
        for point_index in range(total_points):
            for normal_direction_index in range(total_directions):
                lookup[point_index][normal_direction_index] = (
                    get_plane_intercept_by_index(
                        prime, dimension, point_index, normal_direction_index
                    )
                )
        return lookup

    return plane_lookup


def line_lookup_factory(prime, dimension):
    @njit
    def line_lookup():
        total_points = prime**dimension
        total_directions = ((prime**dimension) - 1) // (prime - 1)
        lookup = np.empty((total_points, total_directions), dtype=np.uint16)
        for point_index in range(total_points):
            for direction_index in range(total_directions):
                lookup[point_index][direction_index] = (
                    get_line_intercept_by_index(
                        prime, dimension, point_index, direction_index
                    )
                )
        return lookup

    return line_lookup


@njit
def get_plane_intercept(prime, dimension, point, normal_direction):
    dot_product = 0
    for coord in range(dimension):
        dot_product += point[coord] * normal_direction[coord]
    return dot_product % prime


@njit
def get_line_intercept(prime, dimension, point, direction):
    non_zero_index = dimension - 1
    for i in range(dimension - 1, -1, -1):
        if direction[i] % prime != 0:
            non_zero_index = i
            break
    normalized_direction = np.zeros(dimension, dtype=np.uint8)
    normalizer = pow(direction[non_zero_index], -1, prime)
    for index in range(dimension):
        normalized_direction[index] = (direction[index] * normalizer) % prime
    key_coord = (point[non_zero_index]) % prime

    intercept = np.zeros(dimension - 1, dtype=np.uint8)
    intercept_index = 0
    for index in range(dimension):
        if index == non_zero_index:
            continue
        intercept[intercept_index] = (
            point[index] - (direction[index] * key_coord)
        ) % prime

        intercept_index += 1
    return intercept


@njit
def point_to_index(prime, dimension, point):
    index = 0
    for i in range(dimension - 1, -1, -1):
        index += int(point[i]) * (prime ** (dimension - 1 - i))
    return index


@njit
def index_to_point(prime, dimension, index):
    point = np.zeros(dimension, dtype=np.uint8)
    to_expand = index
    for i in range(dimension - 1, -1, -1):
        point[i] = to_expand % prime
        to_expand //= prime
    return point


@njit
def intercept_to_index(prime, dimension, intercept):
    return point_to_index(prime, dimension - 1, intercept)


@njit
def index_to_intercept(prime, dimension, intercept):
    return index_to_point(prime, dimension - 1, intercept)


@njit
def direction_to_index(prime, dimension, direction):
    non_zero_index = dimension - 1
    for i in range(dimension - 1, -1, -1):
        if direction[i] % prime != 0:
            non_zero_index = i
            break
    shift = ((prime**non_zero_index) - 1) // (prime - 1)
    sub_direction = np.empty(non_zero_index, dtype=np.uint8)
    for i in range(non_zero_index):
        sub_direction[i] = direction[i]
    point_index = point_to_index(prime, non_zero_index, sub_direction)
    return shift + point_index


@njit
def index_to_direction(prime, dimension, index):
    direction = np.zeros(dimension, dtype=np.uint8)

    # could do this more cleverly
    sub_dimension = 0
    threshold = ((prime**sub_dimension) - 1) // (prime - 1)
    while threshold <= index:
        sub_dimension += 1
        threshold = ((prime**sub_dimension) - 1) // (prime - 1)
    sub_dimension -= 1
    shift = ((prime**sub_dimension) - 1) // (prime - 1)
    sub_direction = index_to_point(prime, sub_dimension, index - shift)

    for i in range(sub_dimension):
        direction[i] = sub_direction[i]
    direction[sub_dimension] = 1
    return direction
