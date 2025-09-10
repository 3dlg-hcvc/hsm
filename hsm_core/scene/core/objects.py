from __future__ import annotations
from typing import Optional, Tuple, Any, Dict, List
import os
from dataclasses import dataclass, field
from hsm_core.utils.util import numpy_to_python
from typing import TYPE_CHECKING
from hsm_core.scene.core.objecttype import ObjectType

if TYPE_CHECKING:
    from hsm_core.scene.core.motif import SceneMotif

@dataclass
class Bounds:
    """Represents 2D bounds of a surface."""
    min: Tuple[float, float]  # (x, y) minimum coordinates
    max: Tuple[float, float]  # (x, y) maximum coordinates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": list(self.min),
            "max": list(self.max)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bounds':
        return cls(
            min=tuple(data["min"]),
            max=tuple(data["max"])
        )

@dataclass
class Surface:
    """Represents a surface in a layer."""
    surface_id: int
    area: float
    bounds: Bounds
    center: Tuple[float, float]  # (x, y) center coordinates
    color: str
    depth: float
    width: float
    geometry: Optional[Dict[str, List[List[float]]]] = None  # Optional vertices data
    local_transform: Optional[List[List[float]]] = None # Transformation matrix for the surface

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "surface_id": self.surface_id,
            "area": self.area,
            "bounds": self.bounds.to_dict(),
            "center": list(self.center),
            "color": self.color,
            "depth": self.depth,
            "width": self.width
        }
        if self.geometry:
            data["geometry"] = self.geometry
        if self.local_transform:
            data["local_transform"] = self.local_transform
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Surface':
        return cls(
            surface_id=data["surface_id"],
            area=data["area"],
            bounds=Bounds.from_dict(data["bounds"]),
            center=tuple(data["center"]),
            color=data["color"],
            depth=data["depth"],
            width=data["width"],
            geometry=data.get("geometry"),
            local_transform=data.get("local_transform")
        )

@dataclass
class Layer:
    """Represents a layer in the object's layout."""
    height: float
    is_top_layer: bool
    space_above: float
    surfaces: List[Surface]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "is_top_layer": self.is_top_layer,
            "space_above": self.space_above,
            "surfaces": [surface.to_dict() for surface in self.surfaces]
        }

    def to_gpt_dict(self) -> Dict[str, Any]:
        """Convert layer to simplified format for GPT."""
        return {
            "height": self.height,
            "is_top_layer": self.is_top_layer,
            "space_above": self.space_above,
            "surfaces": [
                {
                    "surface_id": surface.surface_id,
                    "width": surface.width,
                    "depth": surface.depth
                }
                for surface in self.surfaces
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Layer':
        return cls(
            height=data["height"],
            is_top_layer=data["is_top_layer"],
            space_above=data["space_above"],
            surfaces=[Surface.from_dict(surface) for surface in data["surfaces"]]
        )

@dataclass
class LayoutData:
    """Represents the complete layout data for an object."""
    layers: Dict[str, Layer]  # Maps layer names (e.g., "layer_0") to Layer objects

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            layer_name: layer.to_dict()
            for layer_name, layer in self.layers.items()
        }

    def to_gpt_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to simplified format for GPT."""
        return {
            layer_name: layer.to_gpt_dict()
            for layer_name, layer in self.layers.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> 'LayoutData':
        return cls(layers={
            layer_name: Layer.from_dict(layer_data)
            for layer_name, layer_data in data.items()
        })

    @classmethod
    def from_raw_layer_data(cls, raw_layer_data: Dict[str, Any]) -> Dict[str, 'LayoutData']:
        """Convert raw layer data to structured LayoutData objects.
        
        Args:
            raw_layer_data: Dictionary containing raw layer information
            
        Returns:
            dict: Mapping of object names to their LayoutData
        """
        layout_data = {}
        
        for obj_name, layers in raw_layer_data.items():
            layer_dict = {}
            for layer_key, layer_info in layers.items():
                if not layer_key.startswith("layer_"):
                    continue
                    
                # Convert surfaces to Surface objects
                surfaces = []
                for surface in layer_info["surfaces"]:
                    bounds = Bounds(
                        min=tuple(surface["bounds"]["min"]),
                        max=tuple(surface["bounds"]["max"])
                    )
                    surfaces.append(Surface(
                        surface_id=surface["surface_id"],
                        area=surface["area"],
                        bounds=bounds,
                        center=tuple(surface["center"]),
                        color=surface.get("color", "default"),
                        depth=surface["depth"],
                        width=surface["width"],
                        geometry=surface.get("geometry"),
                        local_transform=surface.get("local_transform")
                    ))
                
                # Create Layer object
                layer = Layer(
                    height=layer_info["height"],
                    is_top_layer=layer_info["is_top_layer"],
                    space_above=layer_info["space_above"],
                    surfaces=surfaces
                )
                layer_dict[layer_key] = layer
                
            # Create LayoutData object for this object
            layout_data[obj_name] = cls(layers=layer_dict)
        
        return layout_data

    def get_surface_by_id(self, layer_key: str, surface_id: int) -> Optional[Surface]:
        """Get a surface by its ID from a specific layer.
        
        Args:
            layer_key: Key of the layer to search in
            surface_id: ID of the surface to find
            
        Returns:
            Optional[Surface]: The surface if found, None otherwise
        """
        if layer_key not in self.layers:
            return None
        return next((s for s in self.layers[layer_key].surfaces if s.surface_id == surface_id), None)

    def get_layer_info(self, layer_key: str) -> Optional[Layer]:
        """Get layer information by key.
        
        Args:
            layer_key: Key of the layer to retrieve
            
        Returns:
            Optional[Layer]: The layer if found, None otherwise
        """
        return self.layers.get(layer_key)

    def get_surface_info(self, layer_key: str, surface_id: int) -> Optional[Dict[str, Any]]:
        """Get surface information in a format suitable for collision solving.
        
        Args:
            layer_key: Key of the layer containing the surface
            surface_id: ID of the surface to get info for
            
        Returns:
            Optional[Dict[str, Any]]: Surface information if found, None otherwise
        """
        surface = self.get_surface_by_id(layer_key, surface_id)
        if not surface:
            return None
            
        return {
            "geometry": surface.geometry,
            "bounds": surface.bounds.to_dict(),
            "center": list(surface.center),
            "width": surface.width,
            "depth": surface.depth,
            "area": surface.area
        }

@dataclass
class SceneObject:
    """Class representing an object to be placed in the 3D scene.

    Defines properties needed to place and transform a 3D mesh in world space coordinates.
    World coordinates: X=right, Y=up, Z=forward (negative into screen).
    Objects are centered in XZ plane with Y position at bottom before transforms.
    """
    name: str  # Unique identifier for the object
    position: Tuple[float, float, float]  # (x, y, z) coordinates in world space
    dimensions: Tuple[float, float, float]  # Physical size as (width, height, depth) in meters
    rotation: float  # Rotation in degrees, counterclockwise around Y axis (0° = -Z)
    mesh_path: str  # Path to the 3D mesh file (.glb, .obj, etc.)
    obj_type: ObjectType = ObjectType.UNDEFINED  # Category of object
    child_motifs: List['SceneMotif'] = field(default_factory=list)  # Child motifs if any
    layout_data: Optional[LayoutData] = None  # Layout data if any
    parent_name: Optional[str] = None  # Name of parent object if any
    front_vector: Optional[Tuple[float, float, float]] = None  # Front vector if any
    id: Optional[str] = None  # Optional unique identifier
    motif_id: Optional[str] = None # ID of the motif this object belongs to
    wall_id: Optional[str] = None  # ID of the specific wall this object should be attached to (for WALL objects)
    optimized_world_pos: Optional[Tuple[float, float, float]] = None
    _preprocessing_data: Optional[Dict[str, Any]] = field(default_factory=dict)  # Internal preprocessing data


    def to_dict(self) -> Dict[str, Any]:
        """Convert SceneObject to a dictionary for serialization."""
        data = {
            "name": self.name,
            "position": numpy_to_python(self.position),
            "dimensions": numpy_to_python(self.dimensions),
            "rotation": numpy_to_python(self.rotation),
            "mesh_path": str(self.mesh_path) if self.mesh_path else None,
            "obj_type": self.obj_type.value if isinstance(self.obj_type, ObjectType) else self.obj_type,
            "parent_name": self.parent_name,
            "id": self.id,
            "motif_id": self.motif_id,
            "child_motifs": [m.to_dict() for m in self.child_motifs] if self.child_motifs else [],
            "wall_id": self.wall_id
        }
        if self.layout_data:
            data["layout_data"] = self.layout_data.to_dict()
        if self.front_vector:
            data["front_vector"] = numpy_to_python(self.front_vector)
        if self.optimized_world_pos:
            data["optimized_world_pos"] = numpy_to_python(self.optimized_world_pos)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneObject':
        """Create a SceneObject instance from a dictionary."""
        try:
            # Required fields
            required_fields = ["name", "position", "dimensions", "rotation", "mesh_path", "obj_type"]
            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            # Create instance with required and optional fields
            obj = cls(
                name=data["name"],
                position=tuple(data["position"]),
                dimensions=tuple(data["dimensions"]),
                rotation=data["rotation"],
                mesh_path=data.get("mesh_path") or "",
                obj_type=ObjectType(data["obj_type"]) if data.get("obj_type") else ObjectType.UNDEFINED,
                wall_id=data.get("wall_id"),
                parent_name=data.get("parent_name"),
                id=data.get("id"),
                motif_id=data.get("motif_id"),
                layout_data=LayoutData.from_dict(data["layout_data"]) if data.get("layout_data") else None,
                child_motifs=[SceneMotif.from_dict(child) for child in data.get("child_motifs", [])],
                optimized_world_pos=tuple(data["optimized_world_pos"]) if data.get("optimized_world_pos") else None
            )

            return obj
        except Exception as e:
            raise ValueError(f"Failed to create SceneObject from dictionary: {e} data: {data}")

    def add_transform(self, transform_name: str, transform_info: 'TransformInfo'):
        """Adds a transform to the object's preprocessing data."""
        if self._preprocessing_data is None:
            self._preprocessing_data = {}
        self._preprocessing_data.setdefault('transforms', {})[transform_name] = transform_info

    def get_transform(self, transform_name: str) -> Optional['TransformInfo']:
        """Gets a transform from the object's preprocessing data."""
        transforms = self._preprocessing_data.get('transforms') if self._preprocessing_data else None
        return transforms.get(transform_name) if transforms else None

    def has_hssd_alignment(self) -> bool:
        """
        Check whether the object possesses a valid HSSD alignment transform.
        """
        transform = self.get_transform("hssd_alignment")
        return transform is not None and getattr(transform, "transform_matrix", None) is not None
        
    def get_hssd_alignment_transform(self) -> Optional['TransformInfo']:
        """Get the HSSD alignment transform."""
        return self.get_transform('hssd_alignment')

    def get_mesh_id(self) -> str:
        """Returns the mesh id computed from the object's mesh_path."""
        try:
            return os.path.splitext(os.path.basename(self.mesh_path))[0]
        except Exception:
            return ""

    def to_gpt_dict(self) -> Dict[str, Any]:
        """Convert SceneObject to a simplified dictionary format for GPT."""
        return {
            "id": self.id if self.id else self.name,
            "name": self.name,
            "dimensions": self.dimensions
        }

    @classmethod
    def list_to_gpt_dict(cls, objects: List['SceneObject']) -> List[Dict[str, Any]]:
        """Convert a list of SceneObjects to simplified dictionary format for GPT.
        
        Args:
            objects: List of SceneObject instances
            
        Returns:
            List[Dict[str, Any]]: List of simplified object representations
        """
        return [obj.to_gpt_dict() for obj in objects]

    def get_world_position(self) -> Tuple[float, float, float]:
        """Return the final world-space position, preferring optimized position if available."""
        return self.optimized_world_pos if self.optimized_world_pos is not None else self.position