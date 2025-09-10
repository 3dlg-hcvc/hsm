"""
Export Module

This module handles all input and output operations for scenes, including
saving to various formats and exporting 3D models.
"""

from __future__ import annotations
from pathlib import Path
import traceback

from hsm_core.scene.utils.utils import save_scene_state
from hsm_core.utils.stk_utils import save_stk_scene_state
from hsm_core.utils import get_logger


def save_scene(scene, output_dir: Path, suffix: str = "", recreate_scene: bool = False, save_scene_state: bool = False):
    """
    Save the scene to output directory.

    Args:
        scene: The scene object to save
        output_dir: Directory to save to
        suffix: Suffix for filename
        recreate_scene: Whether to recreate the scene before saving
        save_scene_state: Whether to save scene state (controls multiple saves during pipeline)
    """
    if recreate_scene:
        scene.invalidate_scene()

    if not scene.is_scene_created():
        scene.create_scene()

    glb_filename = "room_scene.glb"
    scene_state_filename = "hsm_scene_state.json"
    stk_state_filename = "stk_scene_state.json"
    if suffix:
        glb_filename = f"room_scene_{suffix}.glb"
        scene_state_filename = f"hsm_scene_state_{suffix}.json"
        stk_state_filename = f"stk_scene_state_{suffix}.json"

    export_scene(scene, output_dir / glb_filename, recreate_scene=False)
    if save_scene_state:
        save_scene_state_to_file(scene, output_dir / scene_state_filename)
    save_scene_stk_state(scene, output_dir / stk_state_filename)


def export_scene(scene, output_path: Path, recreate_scene: bool = False) -> None:
    """
    Export the scene to a GLB file.

    Args:
        scene: The scene object to export
        output_path (Path): The path to export the GLB file.
        recreate_scene (bool): If True, reruns layout and recreates the scene even if self.scene exists.
    """
    if scene.scene is None or recreate_scene:
        scene.create_scene()
    logger = get_logger('hsm_core.scene.io.export')
    logger.info(f"Exporting GLB to {output_path}")
    if scene.scene:
        try:
            scene.scene.export(output_path, file_type='glb')
        except (OSError, AttributeError) as e:
            logger.error(f"Error exporting scene: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning("Scene not created, cannot export GLB")


def save_scene_state_to_file(scene, output_path: Path) -> None:
    """Save the current scene state to a JSON file."""
    all_motifs = scene.get_all_motifs()

    try:
        save_scene_state(
            room_desc=scene.room_description,
            scene_motifs=scene._scene_motifs,
            room_vertices=scene.room_vertices,
            door_location=scene.door_location,
            room_type=scene.room_type,
            filename=str(output_path),
            scene_spec=scene.scene_spec,
            window_location=scene.window_location or None,
            room_details=scene.room_details
        )
        logger.info(f"Scene state saved to {output_path} with {len(all_motifs)} motifs")
    except Exception as e:
        logger.error(f"Error saving scene state: {e}")
        traceback.print_exc()


def normalize_vertices(scene) -> tuple[list[list[float]], float, float]:
    """Normalize room vertices by translating to origin."""
    if hasattr(scene, '_cached_normalized_vertices'):
        return scene._cached_normalized_vertices
    from shapely.geometry import Polygon
    polygon = Polygon(scene.room_vertices)
    minx = polygon.bounds[0]
    miny = polygon.bounds[1]
    normalized_room_vertices = [[v[0] - minx, v[1] - miny] for v in scene.room_vertices]
    scene._cached_normalized_vertices = (normalized_room_vertices, minx, miny)
    return scene._cached_normalized_vertices


def get_normalized_cutouts(scene, minx: float, miny: float) -> tuple[object, object]:
    """Get normalized cutouts (door and windows) relative to origin."""
    if hasattr(scene, '_cached_normalized_cutouts'):
        return scene._cached_normalized_cutouts
    door_cutout = getattr(scene.scene_placer, 'door_cutout', None)
    window_cutouts = getattr(scene.scene_placer, 'window_cutouts', None)
    from hsm_core.scene import Cutout
    # Door
    if door_cutout:
        normalized_door_cutout = Cutout(
            location=(door_cutout.location[0] - minx, door_cutout.location[1] - miny),
            cutout_type=door_cutout.cutout_type,
            width=door_cutout.width,
            height=door_cutout.height
        )
        normalized_door_cutout.closest_wall_index = door_cutout.closest_wall_index
        normalized_door_cutout.projection_on_wall = door_cutout.projection_on_wall
        door_location = normalized_door_cutout
    else:
        normalized_door_location = (
            (scene.door_location[0] - minx, scene.door_location[1] - miny)
            if scene.door_location is not None else None
        )
        door_location = normalized_door_location
    # Windows
    if window_cutouts:
        normalized_window_cutouts = []
        for cutout in window_cutouts:
            normalized_cutout = Cutout(
                location=(cutout.location[0] - minx, cutout.location[1] - miny),
                cutout_type=cutout.cutout_type,
                width=cutout.width,
                height=cutout.height,
                bottom_height=cutout.bottom_height
            )
            normalized_cutout.closest_wall_index = cutout.closest_wall_index
            normalized_cutout.projection_on_wall = cutout.projection_on_wall
            normalized_window_cutouts.append(normalized_cutout)
        window_locations = normalized_window_cutouts
    elif hasattr(scene, 'window_location') and scene.window_location:
        normalized_window_locations = [(w[0] - minx, w[1] - miny) for w in scene.window_location]
        window_locations = normalized_window_locations
    else:
        window_locations = None
    scene._cached_normalized_cutouts = (door_location, window_locations)
    return scene._cached_normalized_cutouts


def save_scene_stk_state(scene, output_path: Path) -> None:
    """Save the scene state in STK format."""
    # Ensure scene is created before accessing scene_placer
    if scene.scene_placer is None:
        if scene.scene is None:
            scene.create_scene()

    if scene.scene_placer:
        stk_objects = [
            (obj["id"], list(obj["position"]), obj["rotation"], obj["transform_matrix"])
            for obj in scene.scene_placer.placed_objects
            if all(k in obj for k in ["id", "position", "rotation", "transform_matrix"])
        ]
    else:
        logger.warning("Could not create scene_placer, saving STK state with empty objects list")
        stk_objects = []

    try:
        normalized_room_vertices, minx, miny = normalize_vertices(scene)
        door_location, window_locations = get_normalized_cutouts(scene, minx, miny)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error normalizing room vertices or cutouts: {e}")
        normalized_room_vertices = scene.room_vertices
        door_location = scene.door_location
        window_locations = scene.window_location if hasattr(scene, 'window_location') else None

    # Extract filename from the path and pass directory separately
    filename = output_path.name
    output_dir = output_path.parent

    save_stk_scene_state(stk_objects, normalized_room_vertices, door_location, output_dir,
                         window_locations=window_locations, filename=filename)
