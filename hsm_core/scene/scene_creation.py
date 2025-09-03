"""Scene creation utilities for converting motifs to 3D scenes."""

import logging
from pathlib import Path
from shapely.geometry import Polygon
import trimesh
import numpy as np
from typing import List, Dict, Tuple, Optional

from hsm_core.constants import FLOOR_HEIGHT, WALL_HEIGHT
from hsm_core.scene.objects import SceneObject
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.motif import SceneMotif
from hsm_core.scene.spec import ObjectSpec
from hsm_core.scene.placer import SceneObjectPlacer
from hsm_core.scene.mesh_utils import preprocess_object_mesh

logger = logging.getLogger(__name__)


def create_scene_from_motifs(
    scene_motifs: List[SceneMotif],
    room_polygon: Optional[Polygon] = None,
    door_location: Tuple[float, float] = (0, 0),
    window_location: Optional[List[Tuple[float, float]]] = None,
    floor_height: float = FLOOR_HEIGHT,
    room_height: float = WALL_HEIGHT,
    enable_spatial_optimization: bool = True,
    scene_manager = None,
    output_dir: Optional[str] = None
) -> Tuple[trimesh.Scene, SceneObjectPlacer]:
    """
    Create a 3D scene from a list of motifs.

    This function processes a list of motifs to create a trimesh scene, including room geometry,
    objects, and their placements. It supports spatial optimization and scene export.

    Args:
        scene_motifs: List of motifs to include in the scene
        room_polygon: Shapely Polygon defining the room shape
        door_location: (x, y) coordinates for the door
        window_location: List of (x, y) coordinates for windows
        floor_height: Height of the floor from origin
        room_height: Height of the room from floor
        enable_spatial_optimization: Whether to enable spatial optimization
        scene_manager: The scene manager instance for accessing scene properties
        output_dir: Directory for saving output files

    Returns:
        trimesh.Scene: The generated 3D scene
    """

    # Initialize scene placer
    scene_placer = SceneObjectPlacer(room_height=room_height)

    # Store scene motifs in placer for optimized coordinate checking
    scene_placer._scene_motifs = scene_motifs

    # Create room geometry
    if room_polygon:
        scene_placer.create_room_geom(room_polygon, door_location, window_location)

    # Build scene context for parent lookups
    scene_context: Dict[int, Tuple[SceneMotif, ObjectSpec]] = {}
    # Create a comprehensive context from all motifs, including children
    all_motifs_for_context = []
    for m in scene_motifs:
        all_motifs_for_context.append(m)
        for obj in m.objects:
            if obj.child_motifs:
                all_motifs_for_context.extend(obj.child_motifs)

    for motif in all_motifs_for_context:
        if hasattr(motif, 'object_specs') and motif.object_specs:
            for obj_spec in motif.object_specs:
                if hasattr(obj_spec, 'id') and obj_spec.id is not None:
                    try:
                        scene_context[int(obj_spec.id)] = (motif, obj_spec)
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping object spec with non-integer ID {obj_spec.id} in motif {motif.id}")

    logger.info(f"Built scene context with {len(scene_context)} object specs for parent lookups")

    # Set scene context on placer for ObjectSpec lookups
    scene_placer.set_scene_context(scene_context)

    # Process all top-level motifs and collect SceneObjects
    all_scene_objects: List[SceneObject] = []

    for motif in scene_motifs:
        # We only process top-level motifs here. Child motifs (small objects)
        # are processed recursively within `create_scene_objects_from_motif`.
        scene_objects = create_scene_objects_from_motif(motif, scene_context=scene_context)
        all_scene_objects.extend(scene_objects)

    print(f"Processed {len(all_scene_objects)} scene objects from all motifs")

    # Place all objects in the scene
    preprocessed_meshes: Dict[str, trimesh.Trimesh] = {}
    for obj in all_scene_objects:
        preprocessed_mesh = preprocess_object_mesh(obj, verbose=False)
        if preprocessed_mesh:
            preprocessed_meshes[obj.name] = preprocessed_mesh

    # Create scene with processed objects
    scene: trimesh.Scene = scene_placer.place_objects(all_scene_objects, preprocessed_meshes)

    # Apply global scene transform to flip along Z-axis for Trimesh camera alignment
    flip_z = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    scene.apply_transform(flip_z)

    return scene, scene_placer


def create_scene_objects_from_motif(
    motif: SceneMotif,
    parent_obj: Optional[SceneObject] = None,
    scene_context: Optional[Dict[int, Tuple[SceneMotif, ObjectSpec]]] = None,
    top_level_motif_id: Optional[str] = None
) -> List[SceneObject]:
    """
    Create SceneObject instances from a SceneMotif.

    This function processes a motif's arrangement and recursively processes
    any child motifs (like small objects) to create a flat list of SceneObjects
    for the entire hierarchy under the given motif.

    Args:
        motif: The motif to process
        parent_obj: The parent SceneObject if this is a child motif
        scene_context: Global context for looking up object specifications
        top_level_motif_id: The ID of the top-level motif in a recursive call.

    Returns:
        A list of all SceneObjects created from this motif and its children.
    """
    # Return optimized objects directly if available
    if motif.is_spatially_optimized and hasattr(motif, '_scene_objects') and motif._scene_objects:
        return list(motif._scene_objects.values())

    # Extract wall ID and parent name
    wall_id = _get_wall_id(motif)
    parent_name = _get_parent_name_for_small_motif(motif, scene_context)
    current_top_level_motif_id = top_level_motif_id or motif.id

    # Preserve child motif links from original objects
    original_objects_by_name = {obj.name: obj for obj in motif.objects}

    # Process objects in the motif's arrangement to get their transforms
    scene_objects = process_arrangement_objects(
        arrangement=motif.arrangement,
        motif_rotation=motif.rotation,
        motif_position=motif.position,
        object_type=motif.object_type,
        wall_id=wall_id,
        parent_name=parent_name,
        motif_id=motif.id  # Use the actual motif ID, not the top-level one
    )

    # Process child motifs and collect all objects
    all_objects_in_motif_and_children = _process_child_motifs(
        scene_objects, motif, scene_context, current_top_level_motif_id
    )

    # Store the direct objects in the motif object itself
    motif.add_objects(scene_objects)

    if parent_obj and motif not in parent_obj.child_motifs:
        parent_obj.child_motifs.append(motif)


    return all_objects_in_motif_and_children


def process_arrangement_objects(arrangement, motif_rotation: float = 0,
                              motif_position: Tuple[float, float, float] = (0, 0, 0),
                              object_type: ObjectType = ObjectType.UNDEFINED,
                              wall_id: Optional[str] = None,
                              parent_name: Optional[str] = None,
                              motif_id: Optional[str] = None,
                              verbose: bool = False) -> List[SceneObject]:
    """
    Process objects in an arrangement to extract their transforms and information.

    Coordinate System:
        - Right-handed coordinate system
        - X: Right direction (positive right)
        - Y: Up direction (positive up)
        - Z: Forward direction (positive out of screen)
        - Rotation: Counterclockwise around Y-axis, 0° faces -Z (south)

    Coordinate convention clarification:
        - The source GLB meshes are modelled with their origin at the *bottom centre*.
        - After local → motif → world transformation we record the **centroid** of the
          mesh in `SceneObject.position`.
        - Down-stream systems (spatial optimiser, renderer, etc.) must therefore obtain
          the true bottom-Y via `position[1] - height/2`.

    Assumptions:
        - Objects are initially aligned with world south (facing -Z)
        - Each object is first transformed by its local transform (no_scale_matrix),
          then by the group (motif) rotation
    """
    scene_objects: List[SceneObject] = []

    # Adjust motif rotation based on object type
    adjusted_motif_rotation = _adjust_motif_rotation(motif_rotation, object_type)

    # Compute the rotation matrix for the group rotation (Y-axis)
    motif_rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(adjusted_motif_rotation), [0, 1, 0]  # Rotate around Y axis anti-clockwise
    )

    if verbose:
        print(f"Original Motif Rotation: {motif_rotation}°")
        print(f"Adjusted Motif Rotation: {adjusted_motif_rotation}°")
        print(f"Rotation matrix:\n{motif_rotation_matrix}")
        print(f"Motif Position: {motif_position}")
        print(f"Processing arrangement objects: {arrangement.description}")

    for obj in arrangement.objs:
        if verbose:
            print(f"\nProcessing object: {obj.label}")

        # Get object's local transform
        local_transform: np.ndarray = (
            obj.bounding_box.no_scale_matrix
            if obj.bounding_box.no_scale_matrix is not None
            else np.eye(4)
        )

        if verbose:
            print(f"Local transform:\n{local_transform}")

        # Calculate world transform
        world_transform = _calculate_world_transform(local_transform, motif_rotation_matrix, motif_position)

        if verbose:
            print(f"World transform:\n{world_transform}")

        # Calculate center position and rotation
        center: np.ndarray = world_transform[0:3, 3]
        rotation_matrix: np.ndarray = world_transform[0:3, 0:3]
        rotation_angle: float = _calculate_rotation_angle(rotation_matrix)

        if verbose:
            print(f"Computed center: {center}")
            print(f"Computed rotation angle (from -Z): {rotation_angle}°")
            print("-"*50)

        # Create SceneObject instance
        scene_obj: SceneObject = SceneObject(
            name=obj.label,
            position=(center[0], center[1], center[2]),
            dimensions=(obj.bounding_box.half_size * 2),  # Convert half_size to full dimensions.
            rotation=rotation_angle,
            mesh_path=obj.mesh_path,
            obj_type=object_type,
            parent_name=parent_name,
            wall_id=wall_id,
            motif_id=motif_id,
            id=obj.id  # Set the ID from the arrangement object
        )

        # Transfer transforms from arrangement object to scene object
        _transfer_transforms_to_scene_object(obj, scene_obj, verbose)

        # Add to scene objects list
        scene_objects.append(scene_obj)
        if verbose:
            print(f"Created scene object: {scene_obj.name} with type {scene_obj.obj_type}")

    return scene_objects


def _get_wall_id(motif: SceneMotif) -> Optional[str]:
    """Extract wall ID from motif if it exists."""
    if motif.object_type == ObjectType.WALL:
        if hasattr(motif, 'wall_alignment_id') and motif.wall_alignment_id:
            return motif.wall_alignment_id
    return None


def _get_parent_name_for_small_motif(motif: SceneMotif,
                                   scene_context: Optional[Dict[int, Tuple[SceneMotif, ObjectSpec]]]) -> Optional[str]:
    """Determine parent name for small object motifs."""
    if motif.object_type != ObjectType.SMALL:
        return None

    # Try to get parent from placement data first
    placement_data = getattr(motif, 'placement_data', None)
    if placement_data and 'parent_object' in placement_data:
        parent_name = placement_data['parent_object']
        logger.debug(f"Small motif {motif.id} assigned to parent {parent_name}")
        return parent_name

    # Fallback to object specs lookup
    if not motif.object_specs or not scene_context:
        return None

    for spec in motif.object_specs:
        if hasattr(spec, 'parent_object') and spec.parent_object:
            parent_id = spec.parent_object
            parent_info = scene_context.get(parent_id)
            if parent_info:
                _, parent_spec = parent_info
                parent_name = parent_spec.name
                logger.debug(f"Small motif {motif.id} assigned to parent ID {parent_id} (name: {parent_name})")
                return parent_name
            else:
                # Fallback to generic name if not found in context
                parent_name = f"parent_{parent_id}"
                logger.debug(f"Small motif {motif.id} assigned to parent ID {parent_id} (fallback name: {parent_name})")
                return parent_name

    return None


def _process_child_motifs(scene_objects: List[SceneObject], motif: SceneMotif,
                         scene_context: Optional[Dict[int, Tuple[SceneMotif, ObjectSpec]]],
                         top_level_motif_id: Optional[str]) -> List[SceneObject]:
    """Process child motifs recursively and return all objects."""
    all_objects = list(scene_objects)

    for obj in scene_objects:
        if obj.child_motifs:
            for child_motif in obj.child_motifs:
                child_scene_objects = create_scene_objects_from_motif(
                    child_motif,
                    parent_obj=obj,
                    scene_context=scene_context,
                    top_level_motif_id=top_level_motif_id
                )
                all_objects.extend(child_scene_objects)

    return all_objects


def _adjust_motif_rotation(motif_rotation: float, object_type: ObjectType) -> float:
    """Adjust motif rotation based on object type."""
    if object_type == ObjectType.LARGE or object_type == ObjectType.WALL:
        # Add 180° to make 0° face south (-Z) instead of north (+Z)
        # Only apply adjustment for specific rotations (0° or 180°)
        if motif_rotation == 0 or motif_rotation == 180:
            return motif_rotation + 180
    return motif_rotation


def _calculate_world_transform(local_transform: np.ndarray, motif_rotation_matrix: np.ndarray,
                              motif_position: Tuple[float, float, float]) -> np.ndarray:
    """Calculate world transform from local transform, motif rotation, and position."""
    # Combine transforms: world = motif rotation * local transform
    world_transform = np.dot(motif_rotation_matrix, local_transform)

    # Add translation from motif position
    position_offset = np.array([motif_position[0], motif_position[1], motif_position[2]]) \
                     if len(motif_position) == 3 \
                     else np.array([motif_position[0], 0, motif_position[1]])
    world_transform[0:3, 3] += position_offset

    return world_transform


def _calculate_rotation_angle(rotation_matrix: np.ndarray) -> float:
    """Calculate rotation angle in XZ plane relative to -Z axis (south-facing default)."""
    # Calculate unnormalized world forward direction (object space forward is +Z)
    unnormalized_front = rotation_matrix[:, 2]
    world_front = unnormalized_front / np.linalg.norm(unnormalized_front)

    # Calculate rotation angle in the XZ plane relative to -Z axis
    return (np.degrees(np.arctan2(world_front[0], world_front[2])) % 360)


def _transfer_transforms_to_scene_object(obj, scene_obj: SceneObject, verbose: bool) -> None:
    """Transfer HSSD and rotation optimization transforms from arrangement object to scene object."""
    # Transfer HSSD transforms
    if obj.has_hssd_alignment():
        hssd_transform_info = obj.get_hssd_alignment_transform()
        if hssd_transform_info:
            scene_obj.add_transform('hssd_alignment', hssd_transform_info)
            if verbose:
                logger.debug(f"Transferred HSSD alignment transform to {scene_obj.name}")

    # Transfer rotation optimisation data
    if hasattr(obj, 'transform_tracker') and obj.transform_tracker is not None and obj.mesh_path:
        try:
            mesh_id_tmp = obj.transform_tracker.get_mesh_id_from_path(obj.mesh_path)
            rot_opt_info = obj.transform_tracker.get_transform_by_type(mesh_id_tmp, "rotation_optimization")
            if rot_opt_info is not None:
                scene_obj.add_transform('rotation_optimization', rot_opt_info)
                if verbose:
                    logger.debug(f"Transferred rotation optimisation transform to {scene_obj.name}")
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to transfer rotation optimisation data for {scene_obj.name}: {e}")
