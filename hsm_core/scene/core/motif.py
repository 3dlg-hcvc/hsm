from typing import Optional, Any, TYPE_CHECKING
from matplotlib.figure import Figure

from hsm_core.vlm.utils import round_nested_values
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.core.spec import ObjectSpec
from hsm_core.utils import get_logger

if TYPE_CHECKING:
    from hsm_core.scene.core.objects import SceneObject
from hsm_core.utils.util import numpy_to_python
from hsm_core.scene_motif.core.arrangement import Arrangement

logger = get_logger('scene.core.motif')

class SceneMotif:
    """Class representing an arrangement of objects in the scene.
    
    A SceneMotif defines a group of related objects that form a functional unit,
    such as a seating area, dining area, etc. It maintains the spatial relationships
    between objects and provides methods to manipulate them as a group.
    
    """
    def __init__(self, id: str, glb_path: Optional[str] = None, extents: tuple = (0, 0, 0), 
                 position: tuple[float, float, float] = (0, 0, 0), rotation: float = 0, description: str = "",
                 fig: Optional[Figure] = None, scene_objects: Optional[list['SceneObject']] = None, 
                 arrangement: Optional[Arrangement] = None, object_specs: Optional[list[ObjectSpec]] = None,
                 object_type: ObjectType = ObjectType.UNDEFINED, ignore_collision: bool = False,
                 parent_id: Optional[str] = None, height_limit: float = -1,
                 wall_alignment: bool = False, wall_alignment_id: Optional[str] = None):
        """
        Args:
            id (str): Unique identifier for the motif
            glb_path (str): Path to GLB file containing the 3D model
            extents (tuple): Physical size as (width, height, depth) in meters
            position (tuple[float, float, float]): (x, y, z) coordinates in world space
            rotation (float): Rotation in degrees around Y axis
            description (str): Text description of the motif
            arrangement (Arrangement): Associated arrangement object
            fig (plt.figure): Associated matplotlib figure
            scene_objects (dict): Dictionary of SceneObject instances
            object_type (ObjectType): Type of the motif
            object_specs (list[ObjectSpec]): List of furniture specifications
            ignore_collision (bool): Whether to ignore collision for this motif
            height_limit (float): Height limit for the motif
            wall_alignment (bool): Whether the motif should be aligned to a wall
            wall_alignment_id (str, optional): The ID of the wall to align to
        """
        self.id: str = id
        self.glb_path: Optional[str] = glb_path
        self.extents: tuple[float, float, float] = extents  # (width, height, depth)
        self.position: tuple[float, float, float] = position  # (x, y, z)
        self.rotation: float = rotation  # degree
        self.description: str = description
        self.arrangement: Optional[Arrangement] = arrangement
        self.fig: Optional[Figure] = fig
        self.object_type: ObjectType = object_type

        self._scene_objects: dict[str, 'SceneObject'] = {}
        self.add_objects(scene_objects or [])
        self.parent_id: Optional[str] = parent_id
        self.object_specs: list[ObjectSpec] = object_specs or []
        self.ignore_collision: bool = ignore_collision
        self.height_limit: float = height_limit
        
        # Wall alignment properties
        self.wall_alignment = wall_alignment
        self.wall_alignment_id = wall_alignment_id
        
        # Spatial optimization tracking
        self.is_spatially_optimized: bool = False
        self.original_position: Optional[tuple[float, float, float]] = None
        self.original_rotation: Optional[float] = None
        
        # Placement data from small object population stage
        self.placement_data: Optional[dict] = None
        
    def add_object(self, obj: 'SceneObject') -> None:
        """Add a single scene object to the motif."""
        self._scene_objects[obj.name] = obj
        
    def add_objects(self, objects: list['SceneObject']) -> None:
        """Add multiple scene objects to the motif."""
        for obj in objects:
            self.add_object(obj)
            
    def get_object(self, name: str) -> Optional['SceneObject']:
        """Get a scene object by name."""
        return self._scene_objects.get(name)
    
    def remove_object(self, name: str) -> None:
        """Remove a scene object by name."""
        self._scene_objects.pop(name, None)
        
    @property
    def scene_objects(self) -> dict[str, 'SceneObject']:
        """Get the scene objects dictionary (read-only)."""
        return self._scene_objects.copy()
    
    @property
    def object_names(self) -> list[str]:
        """Get list of all object names."""
        return list(self._scene_objects.keys())
    
    @property
    def objects(self) -> list['SceneObject']:
        """Get list of all objects."""
        return list(self._scene_objects.values())
    
    def get_objects_by_names(self, object_names: Optional[list[str]] = None) -> tuple[list['SceneObject'], list[str]]:
        """
        Get scene objects by their names. If no names provided, returns all objects.

        Args:
            object_names (list[str], optional): List of object names to retrieve. If None, returns all objects.

        Returns:
            tuple[list['SceneObject'], list[str]]: Tuple containing:
                - List of scene objects matching the given names (or all objects if no names provided)
                - List of object names that were found
        """
        if object_names is None:
            return self.objects, self.object_names
        
        # Create a case-insensitive mapping for object lookup
        case_insensitive_map = {name.lower(): name for name in self._scene_objects.keys()}
        
        # Filter objects that exist in scene_objects and maintain order from object_names
        found_names = []
        found_objects = []
        
        for requested_name in object_names:
            # Try exact match first
            if requested_name in self._scene_objects:
                found_names.append(requested_name)
                found_objects.append(self._scene_objects[requested_name])
            # Fall back to case-insensitive match
            elif requested_name.lower() in case_insensitive_map:
                actual_name = case_insensitive_map[requested_name.lower()]
                found_names.append(actual_name)
                found_objects.append(self._scene_objects[actual_name])
        
        return found_objects, found_names
    
    def __repr__(self):
        furniture_str = ", ".join(f"{spec.name}" for spec in self.object_specs) if self.object_specs else "no furniture"
        return f"SceneMotif(id='{self.id}', pos={self.position}, rot={self.rotation}, furniture=[{furniture_str}])" 
    
    def __str__(self):
        furniture_details = "\n  ".join(
            f"{spec.name} ({spec.dimensions[0]}x{spec.dimensions[1]}x{spec.dimensions[2]})" 
            for spec in self.object_specs
        ) if self.object_specs else "no furniture"
        return f"SceneMotif(id='{self.id}', desc='{self.description}', ext={self.extents}\n  {furniture_details})"
        
    def set_parent(self, parent_object: Optional['SceneObject']) -> None:
        """Set or clear the parent object, updating local transforms."""
        if parent_object == self.parent_object:
            return

        self.parent_object = parent_object



    def get_furniture_by_id(self, furniture_id: int) -> Optional[ObjectSpec]:
        """Get furniture specification by ID.

        Args:
            furniture_id (int): ID of the furniture to find

        Returns:
            Optional[FurnitureSpec]: Matching furniture spec or None if not found
        """
        return next((spec for spec in self.object_specs if spec.id == furniture_id), None)

    def get_child_furniture(self) -> list[ObjectSpec]:
        """Get all child furniture specifications for this motif.

        Returns:
            list[FurnitureSpec]: List of child furniture specifications
        """
        parent_ids = {spec.id for spec in self.object_specs}
        return [
            spec for spec in self.object_specs 
            if spec.parent_object and spec.parent_object in parent_ids
        ]

    def get_mesh_ids(self, object_names: Optional[list[str]] = None) -> dict:
        """
        Retrieve a mapping from object names to their mesh ids.
        If object_names is None, all objects in the motif will be processed.
        """
        objects, _ = self.get_objects_by_names(object_names)
        ids = {}
        for obj in objects:
            try:
                ids[obj.name] = obj.get_mesh_id()
            except Exception as e:
                logger.error(f"Error retrieving mesh id for {obj.name}: {e}")
        return ids

    def to_gpt_dict(self) -> dict[str, Any]:
        return {
            "name": self.id,
            "extents": round_nested_values(self.extents, 2),
            "object_specs": [round_nested_values(spec.to_gpt_dict(), 2) for spec in self.object_specs],
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert SceneMotif to a dictionary representation.
        
        Returns:
            dict: Dictionary containing all relevant SceneMotif data
        """
        scene_objects = []
        for obj in self.objects:
            if not obj:
                logger.warning(f"Skipping None object in motif {self.id}")
                continue

            # Ensure object has required fields
            if not all(hasattr(obj, field) for field in ['name', 'position', 'dimensions', 'rotation', 'mesh_path']):
                logger.warning(f"Object {obj.name if hasattr(obj, 'name') else 'unknown'} missing required fields")
                continue
                
            # Set object ID if not present
            if not obj.id:
                obj.id = f"{self.id}_{obj.name}"
                
            obj_data = numpy_to_python(obj.to_dict())
            scene_objects.append(obj_data)
            
        return {
            "name": self.id,
            "extents": numpy_to_python(self.extents),
            "position": numpy_to_python(self.position),
            "rotation": numpy_to_python(self.rotation),
            "description": self.description,
            "glb_file": str(self.glb_path) if self.glb_path else None,
            "object_type": self.object_type.value,
            "object_specs": [spec.to_dict() for spec in self.object_specs],
            "ignore_collision": self.ignore_collision,
            "scene_objects": scene_objects,
            "wall_alignment": self.wall_alignment,
            "wall_alignment_id": self.wall_alignment_id,
            "is_spatially_optimized": self.is_spatially_optimized
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SceneMotif':
        from hsm_core.scene.core.objects import SceneObject
        
        scene_objects = [SceneObject.from_dict(d) for d in data.get("scene_objects", [])]
        object_specs = [ObjectSpec(**d) for d in data.get("object_specs", [])]
        
        motif = cls(
            id=data["name"], # "name" key is used as id
            extents=tuple(data["extents"]),
            position=tuple(data["position"]),
            rotation=data["rotation"],
            description=data["description"],
            glb_path=data.get("glb_file"),
            scene_objects=scene_objects,
            object_specs=object_specs,
            object_type=ObjectType(data.get("object_type", "undefined")),
            ignore_collision=data.get("ignore_collision", False),
            wall_alignment=data.get("wall_alignment", False),
            wall_alignment_id=data.get("wall_alignment_id")
        )
        
        # Restore optimization state
        motif.is_spatially_optimized = data.get("is_spatially_optimized", False)
        
        return motif

def filter_motifs_by_types(motifs: list['SceneMotif'], object_types: list[ObjectType]) -> list['SceneMotif']:
    """
    Filter a list of SceneMotif objects by their object types.
    
    Args:
        motifs: List of SceneMotif objects to filter
        object_types: List of ObjectType to filter by
        
    Returns:
        List of SceneMotif objects that match the specified object types
    """
    return [motif for motif in motifs if motif.object_type in object_types]

if __name__ == "__main__":
    pass