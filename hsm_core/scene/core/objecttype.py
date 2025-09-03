from __future__ import annotations
from enum import Enum

class ObjectType(Enum):
    WALL = "wall"
    SMALL = "small"
    LARGE = "large"
    CEILING = "ceiling"
    UNDEFINED = "undefined"

    @classmethod
    def from_text(cls, text: str) -> 'ObjectType':
        return cls[text.upper()]
