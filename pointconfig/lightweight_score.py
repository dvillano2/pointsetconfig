import numpy as np
from numba import njit
import pointconfig.lightweight_utils as lwu

PRIME = 11
DIMENSION = 3
plane_lookup_maker = lwu.plane_lookup_factory(PRIME, DIMENSION)
PLANE_LOOKUP = plane_lookup_maker()
line_lookup_maker = lwu.line_lookup_factory(PRIME, DIMENSION)
LINE_LOOKUP = line_lookup_maker()
TOTAL_POINTS = PRIME**3
WORD_LENGTH = TOTAL_POINTS - 4
TOTAL_DIRECTIONS = ((PRIME**DIMENSION) - 1) // (PRIME - 1)
TOTAL_PLANE_INTERCEPTS = PRIME
TOTAL_LINE_INTERCEPTS = PRIME**2

FIXED_INDICES = {0, 1, PRIME, PRIME**2}
VALID_INDICES = np.array(
    [i for i in range(TOTAL_POINTS) if i not in FIXED_INDICES]
)

BATCH_SIZE = 1000


def score_thresholds(PRIME):
    thresholds = {}
    score = 0
    score += PRIME
    thresholds[score] = f"Size a multiple of {PRIME}"
    score += PRIME**2
    thresholds[score] = "Size multiple lies in the correct range"
    score += (1 + PRIME + (PRIME**2)) * (PRIME**2)
    thresholds[score] = f"All planes have {PRIME} or fewer points"
    score += (1 + PRIME + (PRIME**2)) * (PRIME**2) * (PRIME)
    thresholds[score] = "No line contains too many points"
    return thresholds


@njit()
def score_word(word):
    score, size_pass, multiple = _size_scoring(word, PRIME)
    if not size_pass:
        return score
    line_threshold = min(multiple, PRIME - multiple)

    # incidence trackers
    # lines
    line_incidence, directions_determined = _get_line_incidence_structures()
    # planes
    (
        plane_incidence,
        plane_equidistribution,
    ) = _get_plane_incidence_strutures()

    # add in the points that are always there
    true_word = _true_word_from_word(word)

    for point_index in range(TOTAL_POINTS):
        if true_word[point_index] == 0:
            continue
        for direction_index in range(TOTAL_DIRECTIONS):
            # planes
            _update_plane_structures(
                point_index,
                direction_index,
                plane_incidence,
                plane_equidistribution,
                multiple,
            )

            # lines
            _update_line_structures(
                point_index,
                direction_index,
                line_incidence,
                directions_determined,
            )

    # score planes
    score += _score_incidence(
        plane_incidence,
        line_incidence,
        line_threshold,
    )
    return score


@njit()
def score_words(words):
    assert BATCH_SIZE == len(words)
    # work out the dtype here, have to
    scores = np.empty(BATCH_SIZE)
    for word_index in range(BATCH_SIZE):
        # if word_index % 100 == 0:
        #    print(f"Scored {word_index + 1} words")
        scores[word_index] = score_word(words[word_index, :])
    return scores


@njit()
def _size_scoring(word, prime):
    pass_size_section = False
    size = 4 + np.sum(word)
    size_mod_p = size % prime
    score = _multiple_scoring(size_mod_p, prime)
    if score < prime:
        return score, pass_size_section, 0
    relative, line_threshold = _relative_scoring(size, prime)
    score += relative
    if relative == prime**2:
        pass_size_section = True
    return score, pass_size_section, line_threshold


@njit()
def _multiple_scoring(size_mod_p, prime):
    if size_mod_p != 0:
        symmetric_postion = min(size_mod_p, prime - size_mod_p)
        halfway = (prime - 1) // 2
        return halfway - symmetric_postion
    return prime


@njit()
def _relative_scoring(size, prime):
    multiple = size // prime
    prime_squared = prime**2
    if multiple <= 2 or prime - 2 <= multiple:
        return prime_squared - multiple, multiple
    return prime_squared, multiple


@njit()
def _get_line_incidence_structures():
    line_incidence = np.zeros(
        (TOTAL_DIRECTIONS, TOTAL_LINE_INTERCEPTS), dtype=np.uint8
    )
    directions_determined = np.zeros(TOTAL_DIRECTIONS, dtype=np.uint8)
    return line_incidence, directions_determined


@njit()
def _get_plane_incidence_strutures():
    plane_incidence = np.zeros(
        (TOTAL_DIRECTIONS, TOTAL_PLANE_INTERCEPTS), dtype=np.uint8
    )
    plane_equidistribution = np.ones(TOTAL_DIRECTIONS, dtype=np.uint8)
    return plane_incidence, plane_equidistribution


@njit()
def _true_word_from_word(word):
    true_word = np.zeros(TOTAL_POINTS, dtype=np.uint8)
    true_word[0] = 1
    true_word[1] = 1
    true_word[PRIME] = 1
    true_word[PRIME**2] = 1
    for point_index in range(WORD_LENGTH):
        true_word[VALID_INDICES[point_index]] = word[point_index]
    return true_word


@njit()
def _update_plane_structures(
    point_index,
    direction_index,
    plane_incidence,
    plane_equidistribution,
    multiple,
):
    plane_intercept = PLANE_LOOKUP[point_index, direction_index]
    plane_incidence[direction_index][plane_intercept] += 1
    if plane_incidence[direction_index][plane_intercept] > multiple:
        plane_equidistribution[direction_index] = 0


@njit()
def _update_line_structures(
    point_index,
    direction_index,
    line_incidence,
    directions_determined,
):
    line_intercept = LINE_LOOKUP[point_index][direction_index]
    line_incidence[direction_index][line_intercept] += 1
    if line_incidence[direction_index][line_intercept] > 1:
        directions_determined[direction_index] = 1


@njit()
def _score_incidence(
    plane_incidence,
    line_incidence,
    line_threshold,
):
    plane_score = 0
    line_score = 0
    equidistribution_score = 0
    plane_passed = True
    line_passed = True
    for direction in range(TOTAL_DIRECTIONS):
        # planes
        max_plane = 0
        min_plane = TOTAL_LINE_INTERCEPTS
        for intercept in range(TOTAL_PLANE_INTERCEPTS):
            incidence = plane_incidence[direction][intercept]
            # track max and min for equiditribuion
            max_plane = max(incidence, max_plane)
            min_plane = min(incidence, min_plane)
            # add appropriate scores depending on if there's
            # a plane with more that PRIME points
            if incidence <= PRIME:
                plane_score += TOTAL_LINE_INTERCEPTS
            else:
                plane_passed = False
                plane_score += TOTAL_LINE_INTERCEPTS - incidence

        # Add equidistribution score, a single totally
        # equidistributed direction should be more valuable
        # than all of them being very close to zero, but nonzero
        local_equidistribution = PRIME - (max_plane - min_plane)
        if local_equidistribution == PRIME:
            equidistribution_score += PRIME * TOTAL_DIRECTIONS
        else:
            equidistribution_score += local_equidistribution
        # lines
        for intercept in range(TOTAL_LINE_INTERCEPTS):
            incidence = line_incidence[direction][intercept]
            if incidence <= line_threshold:
                line_score += TOTAL_PLANE_INTERCEPTS
            else:
                line_passed = False
                line_score += TOTAL_PLANE_INTERCEPTS - incidence
    score = plane_score
    if not plane_passed:
        return score
    score += line_score
    if not line_passed:
        return score
    return score + equidistribution_score
