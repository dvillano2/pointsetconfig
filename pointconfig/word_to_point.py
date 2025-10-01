from pointconfig.lightweight_utils import index_to_point, index_to_direction
from pointconfig.lightweight_score import TOTAL_DIRECTIONS
from numba import njit


def word_to_point(word, prime, dimension):
    point_set = set()
    for index, in_out in enumerate(word):
        if in_out == "1":
            point = index_to_point(prime, dimension, index)
            point = tuple(int(coord) for coord in point)
            point_set.add(point)
    return point_set


def check_equidistribution(point_set, prime, dimension):
    equidistributed_directions = set()
    for direction_index in range(TOTAL_DIRECTIONS):
        direction = index_to_direction(prime, dimension, direction_index)
        tracker = [0] * prime
        for point in point_set:
            dot_product = (
                sum(
                    dir_coord * point_coord
                    for dir_coord, point_coord in zip(direction, point)
                )
                % prime
            )
            tracker[dot_product] += 1
        if all(point_count == tracker[0] for point_count in tracker):
            hashable_direction = tuple(direction.tolist())
            equidistributed_directions.add(hashable_direction)
    return equidistributed_directions
