import numpy as np
from numba import njit


@njit
def sift_heap(prime, normal_direction, plane, heap_pairs, heap_positions):
    while True:
        print(heap_pairs)
        print(heap_positions)
        plane_heap_position = heap_positions[normal_direction, plane]
        parent_0_position = (plane_heap_position * 2) + 1
        parent_1_position = (plane_heap_position * 2) + 2
        if parent_0_position > prime - 1:
            return
        if parent_1_position > prime - 1:
            min_parent_position = parent_0_position
        else:
            parent_0_value = heap_pairs[normal_direction, parent_0_position, 0]
            parent_1_value = heap_pairs[normal_direction, parent_1_position, 0]
            if parent_0_value <= parent_1_value:
                min_parent_position = parent_0_position
            else:
                min_parent_position = parent_1_position

        if (
            heap_pairs[normal_direction, plane_heap_position, 0]
            <= heap_pairs[normal_direction, min_parent_position, 0]
        ):
            return

        min_parent_plane = heap_pairs[normal_direction, min_parent_position, 1]
        print(min_parent_plane)

        # swap heap entries
        tmp0 = heap_pairs[normal_direction, plane_heap_position, 0]
        tmp1 = heap_pairs[normal_direction, plane_heap_position, 1]
        heap_pairs[normal_direction, plane_heap_position, 0] = heap_pairs[
            normal_direction, min_parent_position, 0
        ]
        heap_pairs[normal_direction, plane_heap_position, 1] = heap_pairs[
            normal_direction, min_parent_position, 1
        ]
        heap_pairs[normal_direction, min_parent_position, 0] = tmp0
        heap_pairs[normal_direction, min_parent_position, 1] = tmp1

        # swap location in position tracker
        tmp = heap_positions[normal_direction, plane]
        heap_positions[normal_direction, plane] = min_parent_position
        heap_positions[normal_direction, int(min_parent_plane)] = tmp


@njit
def increment_plane(
    prime, normal_direction, plane, heap_pairs, heap_positions
):
    plane_position = heap_positions[normal_direction, plane]
    heap_pairs[normal_direction, plane_position, 0] += 1
    sift_heap(prime, plane, heap_pairs, heap_positions)
