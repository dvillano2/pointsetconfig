"""
Useful types
"""

from typing import Dict
from typing import Tuple
from typing import Union
from typing import Set

PointType = Tuple[int, ...]
DirectionType = Tuple[int, ...]
PairOfPointsType = Tuple[PointType, PointType]
PlaneIncidenceType = Dict[Tuple[int, ...], Dict[int, int]]
LineIncidenceType = Dict[Tuple[int, ...], Dict[Tuple[int, ...], int]]
LookupEntryType = Dict[str, Dict[Tuple[int, ...], Union[int, Tuple[int, ...]]]]
