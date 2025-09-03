from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict, Iterator
import numpy as np

@dataclass
class ObjectSpec:
    id: int
    name: str
    description: str
    dimensions: List[float]  # [width, height, depth]
    amount: int
    parent_object: Optional[int] = None
    placement_layer: Optional[str] = None
    placement_surface: Optional[int] = None
    wall_id: Optional[int] = None
    required: bool = False # Whether the object is required from the input description
    is_parent: bool = False # Whether the object is a parent object
    height_limit: Optional[float] = None # Maximum height of the object
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ObjectSpec to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectSpec":
        return cls(**data)

    def to_gpt_furniture_info(self) -> Dict[str, Any]:
        """Convert ObjectSpec to dictionary format for GPT."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }

    def to_gpt_dict(self) -> Dict[str, Any]:
        """Convert ObjectSpec to dictionary format for GPT."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dimensions": self.dimensions,
            "amount": self.amount
        }

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Make ObjectSpec iterable like a dict."""
        yield from self.to_dict().items()

    def to_obj(self) -> 'Obj':
        """Convert ObjectSpec to an Obj instance.
        
        Returns:
            Obj: A new Obj instance with properties from this ObjectSpec
        """
        from hsm_core.scene_motif.core.obj import Obj
        from hsm_core.scene_motif.core.bounding_box import BoundingBox

        dimensions = np.array(self.dimensions)
        half_size = dimensions / 2
        bbox = BoundingBox(
            centroid=np.zeros(3),
            half_size=half_size,
            coord_axes=np.eye(3)
        )
        
        return Obj(
            label=self.name.lower().replace(" ", "_"),
            description=self.description,
            bounding_box=bbox,
            mesh=None,
            transform_matrix=np.eye(4),
            mesh_path=None
        )