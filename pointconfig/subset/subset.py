"""
Class for subsets of vector spaces over integers
mod a prime that keeps track of direction
and incidence information.
"""

# pylint: disable=protected-access
import itertools
from typing import Dict
from typing import Optional
from typing import Set
from typing import Tuple

import pointconfig.utils as pcutils
from pointconfig.ambientspace import AmbientSpace
from pointconfig.config_types import DirectionType
from pointconfig.config_types import LineIncidenceType
from pointconfig.config_types import LookupEntryType
from pointconfig.config_types import PairOfPointsType
from pointconfig.config_types import PlaneIncidenceType
from pointconfig.config_types import PointType


class Subset:
    """
    Class for subsets in vectors spaces over the integers
    mod a prime. Keeps track of direction and incidence information.
    """

    # MAKE THE KEYS AMBIENT SPACES
    _LOOKUP: Dict[Tuple[int, int], Dict[PointType, LookupEntryType]] = {}

    @classmethod
    def update_lookup(cls, prime: int, dimension: int) -> bool:
        """
        Checks if key is in lookup table, if not, adds it and
        returns True. Otherwise returns False.
        """
        key = (prime, dimension)
        if key not in cls._LOOKUP:
            cls._LOOKUP[key] = cls.create_new_key(prime, dimension)
            return True
        return False

    @staticmethod
    def create_new_key(
        prime: int, dimension: int
    ) -> Dict[PointType, LookupEntryType]:
        """
        Returns the entry corresponding to a given prime
        and dimension.
        This can be an expensive computation;
        dimension should be small.
        """
        return pcutils.create_full_lookup_table(prime, dimension)

    def __init__(self, prime: int, dimension: int) -> None:
        self.lookup_update_required = Subset.update_lookup(prime, dimension)
        self.space = AmbientSpace(prime, dimension)
        self.points: Set[PointType] = set()
        self.point_pairs_per_direction: Dict[
            DirectionType, Set[PairOfPointsType]
        ] = {}
        self.directions_per_point: Dict[
            PointType, Dict[PointType, DirectionType]
        ] = {}
        self.plane_incidence: PlaneIncidenceType = {
            normal_direction: {intercept: 0 for intercept in range(prime)}
            for normal_direction in pcutils.get_directions(prime, dimension)
        }
        self.line_incidence: LineIncidenceType = {
            direction: {
                intercept: 0
                for intercept in itertools.product(
                    range(prime), repeat=dimension - 1
                )
            }
            for direction in pcutils.get_directions(prime, dimension)
        }

    def get_lookup(self) -> Dict[PointType, LookupEntryType]:
        """
        Easy way to access the lookup table without touching _LOOKUP.
        """
        key = (self.space.prime, self.space.dimension)
        if key not in Subset._LOOKUP:
            raise KeyError(
                f"The key f{key} is not in the shared lookup table yet."
            )
        return Subset._LOOKUP[key]

    def _normalize_direction(self, direction: DirectionType) -> DirectionType:
        """
        Normalizes directions so that the last nonzero entry is always 1.
        """
        index = -1
        while direction[index] == 0:
            index -= 1
        normalizer: int = pow(direction[index], -1, self.space.prime)
        return tuple(
            (coord * normalizer) % self.space.prime for coord in direction
        )

    @staticmethod
    def _get_pair_from_points(
        point_0: PointType, point_1: PointType
    ) -> PairOfPointsType:
        """
        From two points returns their tuple, ordered so that the smaller
        one is always first.
        """
        if point_0 == point_1:
            raise ValueError("Two points in a point pair must be distinct")
        if point_0 < point_1:
            return (point_0, point_1)
        return (point_1, point_0)

    def add_point(self, point: PointType) -> None:
        """
        Adds a point (mod p) from the subset and updates everything.
        """
        if len(point) != self.space.dimension:
            raise ValueError("Length of point must match dimension")
        point = tuple(coord % self.space.prime for coord in point)
        if point in self.points:
            return
        # track the direction information
        # This can be made more clever and cut out some computation
        if self.points:
            self.directions_per_point[point] = {}
        for other_point in self.points:
            direction = tuple(
                (point_coord - other_point_coord) % self.space.prime
                for point_coord, other_point_coord in zip(point, other_point)
            )
            normalized_direction: DirectionType = self._normalize_direction(
                direction
            )
            point_pair: PairOfPointsType = Subset._get_pair_from_points(
                point, other_point
            )
            assert point_pair[0] < point_pair[1]
            # Might be worth a rewrite
            if normalized_direction not in self.point_pairs_per_direction:
                self.point_pairs_per_direction[normalized_direction] = set()
            self.point_pairs_per_direction[normalized_direction].add(
                point_pair
            )

            # Add point to direction
            self.directions_per_point[point][
                other_point
            ] = normalized_direction
            if other_point not in self.directions_per_point:
                self.directions_per_point[other_point] = {}
            self.directions_per_point[other_point][
                point
            ] = normalized_direction

        # update plane indicence
        for normal_direction, intercept in self.get_lookup()[point][
            "planes"
        ].items():
            assert isinstance(intercept, int)
            self.plane_incidence[normal_direction][intercept] += 1
        # update line incidence
        for direction, intercept in self.get_lookup()[point]["lines"].items():
            assert isinstance(intercept, tuple)
            self.line_incidence[direction][intercept] += 1

        self.points.add(point)

    def remove_point(self, point: PointType) -> None:
        """
        Removes a point (mod p)from the subset and updates everything.
        """
        if len(point) != self.space.dimension:
            raise ValueError("Length of point must match dimension")
        point = tuple(coord % self.space.prime for coord in point)
        if point not in self.points:
            raise KeyError("Point in not in the subset.")
        self.points.remove(point)
        # track the direction information
        for other_point in self.points:
            direction = self.directions_per_point[point][other_point]
            point_pair = Subset._get_pair_from_points(point, other_point)
            # remove the point pair
            self.point_pairs_per_direction[direction].remove(point_pair)
            # if direction is no longer determined, get rid of the direction
            if not self.point_pairs_per_direction[direction]:
                del self.point_pairs_per_direction[direction]
            # get rid of the deleted point by point references
            del self.directions_per_point[other_point][point]
            if not self.directions_per_point[other_point]:
                del self.directions_per_point[other_point]
        if point in self.directions_per_point:
            del self.directions_per_point[point]

        # update plane indicence
        for normal_direction, intercept in self.get_lookup()[point][
            "planes"
        ].items():
            assert isinstance(intercept, int)
            self.plane_incidence[normal_direction][intercept] -= 1
        # update line incidence
        for direction, intercept in self.get_lookup()[point]["lines"].items():
            assert isinstance(intercept, tuple)
            self.line_incidence[direction][intercept] -= 1

    @property
    def max_line_incidence(self) -> int:
        """
        Returns the maximum number of points on any line.
        """
        return max(
            (
                value
                for direction in self.line_incidence
                for value in self.line_incidence[direction].values()
            ),
            default=0,
        )

    @property
    def line_incidence_threshold(self) -> Optional[int]:
        """
        Returns the threshold for maximum points on a given line.
        If the subset does not have size that is a muliple of
        the prime, returns None.
        """
        if self.prime_multiple is None:
            return None
        return min(self.prime_multiple, self.space.prime - self.prime_multiple)

    @property
    def below_line_incidence_threshold(self) -> Optional[bool]:
        """
        Returns True if the subset is below the maximum line
        incidence threshold.
        If the subset does not have size that is a muliple of
        the prime, returns None.
        """
        if self.line_incidence_threshold is None:
            return None
        return self.max_line_incidence <= self.line_incidence_threshold

    @property
    def equidistributed_planes(self) -> Set[DirectionType]:
        """
        Returns the normal directions corresponding to families of
        parallel planes over which the subset is equidistributed.
        """
        target = self.prime_multiple
        if target is None:
            return set()
        return set(
            normal_direction
            for normal_direction in self.plane_incidence
            if all(
                value == target
                for value in self.plane_incidence[normal_direction].values()
            )
        )

    @property
    def number_equidistributed_planes(self) -> int:
        """
        Returns the number of families of parallel planes
        over which the subset is equidistributed.
        """
        return len(self.equidistributed_planes)

    @property
    def size(self) -> int:
        """
        Returns the size of the subset.
        """
        return len(self.points)

    @property
    def prime_multiple(self) -> Optional[int]:
        """
        Returns None if the size does not divide the prime.
        Otherwise returns the size divided by the prime.
        """
        if self.size % self.space.prime != 0:
            return None
        return self.size // self.space.prime

    @property
    def directions_determined(self) -> Set[DirectionType]:
        """
        Returns all the directions determined.
        """
        return set(self.point_pairs_per_direction.keys())

    @property
    def number_of_directions_determined(self) -> int:
        """
        Returns the all the directions determines.
        """
        return len(self.point_pairs_per_direction)
