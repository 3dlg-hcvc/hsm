"""Object placement utilities for 3D scenes."""

from pathlib import Path
import traceback
from shapely.geometry import Polygon
import trimesh
import numpy as np
from typing import List, Dict, Tuple, Optional
import shapely.affinity

from hsm_core.constants import FLOOR_HEIGHT, WALL_HEIGHT
from hsm_core.scene.core.objects import SceneObject
from hsm_core.scene.utils.mesh_utils import preprocess_object_mesh
from hsm_core.scene.geometry.room_geometry import create_floor_mesh, create_wall_meshes
from hsm_core.scene.utils.utils import room_to_world, create_front_arrow

from hsm_core.utils import get_logger
logger = get_logger('scene.geometry.placer')


class SceneObjectPlacer:
    """Class handling object placement in 3D scenes."""
    def __init__(self, floor_height: float = FLOOR_HEIGHT, room_height: float = WALL_HEIGHT):
        self.scene: trimesh.Scene = trimesh.Scene()
        self.floor_height: float = floor_height
        self.room_height: float = room_height
        self.placed_objects: List[Dict] = []
        self.scene_offset: List[float] = [0, 0, 0]
        self.door_location: Tuple[float, float] = (0, 0)  # Default door location
        self.door_cutout = None  # Store door cutout object
        self.window_cutouts = None  # Store window cutout objects

    def create_room_geom(self, room_polygon: Polygon, door_location: Tuple[float, float] = (0, 0),
                         window_location: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Tuple[trimesh.Trimesh, np.ndarray]]:
        """Create and add floor and walls to scene.

        Args:
            room_polygon: Room boundary as a shapely Polygon
            door_location: (x, y) position of the door
            window_location: List of (x, y) positions for windows

        Returns:
            Dictionary mapping geometry names to (mesh, transform) tuples
        """
        # TODO: all room gemetry do not have the correct origin
        room_geometry = {}
        # Normalize room polygon to have bottom-left at (0,0)
        bounds = room_polygon.bounds
        normalized_polygon = shapely.affinity.translate(
            room_polygon,
            xoff=-bounds[0],
            yoff=-bounds[1]
        )

        # Translate door location to match the normalized polygon
        normalized_door_location: Tuple[float, float] = (door_location[0] - bounds[0], door_location[1] - bounds[1])
        self.door_location = normalized_door_location

        # Normalize window locations if provided
        normalized_window_location = None
        if window_location:
            normalized_window_location = [(w[0] - bounds[0], w[1] - bounds[1]) for w in window_location]

        logger.info("Creating Room Geometry")
        logger.info("="*50)
        logger.debug(f"Room polygon: {room_polygon.exterior.xy}")
        # logger.debug(f"Normalized polygon bounds: {normalized_polygon.bounds}")
        logger.debug(f"Door location: {self.door_location}")
        if normalized_window_location:
            logger.debug(f"Window locations ({len(normalized_window_location)}): {normalized_window_location}")
        else:
            logger.debug("No windows")
        logger.debug("="*50)

        # Add floor and walls using normalized polygon
        floor_mesh, floor_translation_matrix = create_floor_mesh(normalized_polygon, self.scene_offset)
        room_geometry["floor"] = (floor_mesh, floor_translation_matrix)

        wall_meshes = create_wall_meshes(
            room_polygon=normalized_polygon,
            scene_offset=self.scene_offset,
            door_location=self.door_location,  # now normalized door location
            floor_height=self.floor_height,
            window_location=normalized_window_location,
            scene_placer=self  # Pass self reference to store cutout objects
        )

        for name, mesh in wall_meshes:
            room_geometry[name] = (mesh, np.eye(4))  # Identity transform since walls are pre-transformed
            self.scene.add_geometry(mesh, geom_name=name)

        floor_mesh, floor_transform = room_geometry["floor"]
        self.scene.add_geometry(floor_mesh, geom_name="floor", transform=floor_transform)

        return room_geometry

    def preprocess_object_mesh(self, obj: SceneObject, verbose: bool = True) -> Optional[trimesh.Trimesh]:
        """
        Preprocess object mesh by loading, applying HSSD transforms, and normalizing.

        Args:
            obj: SceneObject to preprocess
            verbose: Whether to print debug information

        Returns:
            Preprocessed trimesh object or None if preprocessing fails
        """
        return preprocess_object_mesh(obj, verbose)

    def place_object(self, obj: SceneObject, verbose: bool = False, preprocessed_mesh: Optional[trimesh.Trimesh] = None) -> bool:
        """Place a single object in the scene."""
        try:
            logger.debug(f"{'='*50}")
            logger.info(f"Placing: {obj.name} ({obj.obj_type.name})")
            logger.debug(f"Dimensions (W,H,D): {obj.dimensions}")

            if obj.child_motifs:
                for motif in obj.child_motifs:
                    logger.debug(f"  Child object: {motif.id}")
            # logger.debug(f"Placement Position: {obj.position} Initial Rotation: {obj.rotation}°")

            # Use preprocessed mesh if provided, otherwise preprocess now
            if preprocessed_mesh is not None:
                loaded_obj = preprocessed_mesh.copy()
            else:
                loaded_obj = preprocess_object_mesh(obj)
                if loaded_obj is None:
                    return False

            world_position = _determine_world_position(obj, self)  # Final position after all transformations

            # Rotate the object around the Y-axis
            rotation_angle = obj.rotation % 360
            y_rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=np.radians(rotation_angle),
                direction=[0, 1, 0],
                point=[0, 0, 0]
            )

            translation_matrix = trimesh.transformations.translation_matrix(world_position)
            logger.debug(f"Final Scene Position: ({world_position[0]:.2f}, {world_position[1]:.2f}, {world_position[2]:.2f}) Final Rotation: {rotation_angle}°")

            final_transform = translation_matrix @ y_rotation_matrix
            self.scene.add_geometry(loaded_obj, geom_name=obj.name, transform=final_transform)
            # _add_front_arrow(self, obj, world_position)

            placed_obj = _create_placed_object_info(obj, world_position)
            self.placed_objects.append(placed_obj)

            return True

        except Exception as e:
            logger.error(f"Error placing {obj.name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def place_objects(self, objects: List[SceneObject], preprocessed_meshes: Optional[Dict[str, trimesh.Trimesh]] = None) -> trimesh.Scene:
        """Place multiple objects in the scene."""
        for obj in objects:
            preprocessed_mesh = preprocessed_meshes.get(obj.name) if preprocessed_meshes else None
            self.place_object(obj, preprocessed_mesh=preprocessed_mesh)
        return self.scene


def _determine_world_position(obj: SceneObject, scene_placer: SceneObjectPlacer) -> np.ndarray:
    """Determine the world position for an object considering various coordinate systems."""
    # Check if object comes from spatially optimized motif (position already in world coordinates)
    if hasattr(obj, 'motif_id') and obj.motif_id:
        parent_motif = None
        # Find the parent motif for this object
        for scene_motif in getattr(scene_placer, '_scene_motifs', []):
            if scene_motif.id == obj.motif_id:
                parent_motif = scene_motif
                break

        # If motif went through spatial optimization, use optimized position directly
        if parent_motif and getattr(parent_motif, 'is_spatially_optimized', False):
            world_position = np.array(obj.position)  # Already in world coordinates
            logger.debug(f"Using spatially optimized world coordinates for {obj.name}")
            return world_position

    return room_to_world(obj.position, scene_placer.scene_offset)

def _add_front_arrow(scene_placer: SceneObjectPlacer, obj: SceneObject, world_position: np.ndarray) -> None:
    """Add front arrow visualization for the object."""
    angle_rad = np.radians(obj.rotation)
    front_vector = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
    obj.front_vector = front_vector.tolist()

    arrow_mesh = create_front_arrow(front_vector, length=0.5, thickness=0.05)
    arrow_translation = trimesh.transformations.translation_matrix(world_position)
    scene_placer.scene.add_geometry(arrow_mesh, geom_name=f"{obj.name}_front_arrow", transform=arrow_translation)

def _create_placed_object_info(obj: SceneObject, world_position: np.ndarray) -> Dict:
    """Create dictionary with placed object information."""
    rot_opt_mat = getattr(obj, '_preprocessing_data', {}).get('rotation_optimization_matrix')
    return {
        "name": obj.name,
        "dimensions": obj.dimensions,
        "position": world_position,
        "rotation": obj.rotation,
        "mesh_path": str(obj.mesh_path),
        "id": Path(obj.mesh_path).stem,
        "transform_matrix": rot_opt_mat if rot_opt_mat is not None else None,
    }
