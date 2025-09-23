"""
Small dataclass for organizing ambient space info
"""
from dataclasses import dataclass


@dataclass
class AmbientSpace:
    """Stores the prime and dimension"""
    prime: int
    dimension: int
