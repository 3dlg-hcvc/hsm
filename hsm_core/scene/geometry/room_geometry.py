"""Room geometry creation utilities for 3D scenes."""

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import trimesh
import numpy as np
from typing import List, Tuple, Optional

from hsm_core.constants import WALL_HEIGHT, WALL_THICKNESS, FLOOR_HEIGHT, DOOR_HEIGHT, DOOR_WIDTH
from hsm_core.scene.geometry.cutout import validate_and_place_cutouts, apply_cutouts_to_wall
from hsm_core.utils import get_logger

logger = get_logger('scene.geometry.room')


def create_floor_mesh(room_polygon: Polygon, scene_offset: List[float]) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Create a trimesh mesh from the room polygon to use as the floor.
    The floor should be in the XZ plane with Y being up.

    Args:
        room_polygon: Shapely Polygon representing room shape
        scene_offset: Translation offset for scene coordinates

    Returns:
        Tuple of (trimesh.Trimesh, transformation_matrix): Floor mesh and its transform
    """
    vertices = np.array(room_polygon.exterior.coords[:-1])
    vertices_2d = vertices.flatten()

    import earcut.earcut as ec
    triangles = ec.earcut(vertices_2d)
    faces = np.array(triangles).reshape(-1, 3)

    # Create vertices in XZ plane (Y up)
    vertices_3d = np.zeros((len(vertices), 3))
    vertices_3d[:, 0] = vertices[:, 0]   # X stays X
    vertices_3d[:, 2] = vertices[:, 1]   # Y becomes positive Z (removed negation)
    vertices_3d[:, 1] = 0                # Y (up) is zero for floor

    mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces)
    mesh.fix_normals()

    # Flip the mesh if normals are pointing down
    if mesh.face_normals[0][1] < 0:  # Check if Y component of normal is negative
        mesh.faces = np.fliplr(mesh.faces)  # Reverse vertex order in faces

    translation_matrix = trimesh.transformations.translation_matrix(scene_offset)
    
    return mesh, translation_matrix


def create_wall_meshes(
    room_polygon: Polygon,
    scene_offset: List[float],
    door_location: Tuple[float, float],
    window_location: Optional[List[Tuple[float, float]]] = None,
    wall_height: float = WALL_HEIGHT,
    wall_thickness: float = WALL_THICKNESS,
    door_height: float = DOOR_HEIGHT,
    door_width: float = DOOR_WIDTH,
    floor_height: float = FLOOR_HEIGHT,
    scene_placer: Optional['SceneObjectPlacer'] = None
) -> List[Tuple[str, trimesh.Trimesh]]:
    """
    Create wall meshes for the room with door and window cutouts.

    Args:
        room_polygon: Room boundary as a shapely Polygon
        scene_offset: Offset for scene coordinates
        door_location: (x, y) position of the door
        window_location: List of (x, y) positions of windows
        wall_height: Height of the walls
        wall_thickness: Thickness of the walls
        door_height: Height of the door
        door_width: Width of the door
        floor_height: Height of the floor
        scene_placer: Optional SceneObjectPlacer to store cutout objects

    Returns:
        List of (name, mesh) tuples for wall geometry
    """
    wall_meshes: List[Tuple[str, trimesh.Trimesh]] = []
    wall_segments = list(zip(room_polygon.exterior.coords[:-1], room_polygon.exterior.coords[1:]))

    # Place door and window cutouts
    door, windows, all_cutouts = validate_and_place_cutouts(
        room_polygon, door_location, window_location, door_width, door_height,
        try_alternative_walls=False
    )

    if scene_placer:
        scene_placer.door_cutout = door if door.is_valid else None
        if windows:
            scene_placer.window_cutouts = windows

    # logger.debug(f"Valid cutouts: {len(all_cutouts)} ({len([c for c in all_cutouts if c.cutout_type == 'door'])} doors, {len(windows)} windows)")

    if window_location and len(windows) < len(window_location):
        logger.warning(f"Only placed {len(windows)} of {len(window_location)} windows")
        logger.info("="*50)

    # Process each wall segment
    for i, (start, end) in enumerate(wall_segments):
        start_arr: np.ndarray = np.array([start[0], 0, start[1]], dtype=float)
        end_arr: np.ndarray = np.array([end[0], 0, end[1]], dtype=float)
        wall_length: float = np.linalg.norm(end_arr - start_arr)
        wall_vector: np.ndarray = end_arr - start_arr
        wall_angle: float = np.arctan2(wall_vector[2], wall_vector[0])
        mid_point: np.ndarray = (start_arr + end_arr) / 2

        # Compute outward normal in XZ plane (right-hand rule)
        wall_dir: np.ndarray = wall_vector / np.linalg.norm(wall_vector)
        normal: np.ndarray = np.array([-wall_dir[2], 0, wall_dir[0]])  # Cross product with Y

        # Create wall box [length, height, thickness]
        wall_box: trimesh.Trimesh = trimesh.creation.box(
            extents=[wall_length, wall_height, wall_thickness]
        )
        wall_box.visual.face_colors = [200, 200, 200, 255]

        # Rotate wall to align with segment
        rot_matrix: np.ndarray = trimesh.transformations.rotation_matrix(
            angle=wall_angle,
            direction=[0, 1, 0]  # WORLD_UP
        )
        wall_box.apply_transform(rot_matrix)

        # Position wall in world space
        translation: np.ndarray = (
            mid_point +                          # Base position
            scene_offset +                       # Scene offset
            np.array([0, wall_height / 2, 0]) +  # Lift to half height
            normal * (wall_thickness / 2)        # Offset by half thickness
        )
        trans_matrix: np.ndarray = trimesh.transformations.translation_matrix(translation)
        wall_box.apply_transform(trans_matrix)
        wall_box = apply_cutouts_to_wall(
            wall_box, i, door, windows, mid_point, wall_dir, wall_length,
            wall_height, wall_thickness, trans_matrix, rot_matrix, room_polygon
        )

        wall_meshes.append((f"wall_{i}", wall_box))

    return wall_meshes

def create_custom_room(vertices):
    """
    Create a custom room mesh from the room polygon.
    """
    # Build polygon then ensure CCW orientation (sign=1.0 => area > 0)
    return orient(Polygon(vertices), sign=1.0)