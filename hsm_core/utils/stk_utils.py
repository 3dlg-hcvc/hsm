import json
import numpy as np

from hsm_core.constants import WALL_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_BOTTOM_HEIGHT, DOOR_WIDTH, DOOR_HEIGHT
from hsm_core.scene.geometry.cutout import Cutout
from hsm_core.utils.util import numpy_to_python
from hsm_core.utils import get_logger

def sv_fix_coordinates(raw_scene: dict) -> dict:
    # match coordinate system for sceneeval
    FIX_MATRIX = np.asarray([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    new_scene = raw_scene.copy()
    
    for obj in new_scene["scene"]["object"]:
        original_matrix = np.asarray(obj["transform"]["data"]).reshape((4, 4), order="F")
        new_matrix = np.dot(FIX_MATRIX, original_matrix)
        obj["transform"]["data"] = new_matrix.reshape(-1, order="F").tolist()
        
    for element in new_scene["scene"]["arch"]["elements"]:
        original_points = np.asarray(element["points"])
        new_points = original_points[:, [0, 2, 1]]
        new_points[:, 0] *= -1
        element["points"] = new_points.tolist()
    
    return new_scene

def create_transform_matrix(position: list[float], rotation: float, object_transform: np.ndarray | None) -> dict:
    """
    Create a 4x4 transform matrix from position, rotation, and optional object-specific transform.
    
    Args:
        position: [x,y,z] coordinates in right-handed coordinate system
        rotation: Angle in degrees, counterclockwise around Y-axis when viewed from above
        object_transform: Optional 4x4 object-specific transformation matrix
        
    Returns:
        dict: Transform matrix in SSTK format with flipped X-axis coordinate
    """
    # Flip X coordinate for position
    position_flipped = [-position[0], position[1], position[2]]
    
    # Create Y-axis rotation matrix
    rotation_rad = np.deg2rad(rotation)
    rot_matrix = np.array([
        [np.cos(rotation_rad), 0, -np.sin(rotation_rad)],
        [0, 1, 0],
        [np.sin(rotation_rad), 0, np.cos(rotation_rad)]
    ])
    
    # Create 4x4 transform matrix
    transform = np.eye(4)
    
    if object_transform is not None:
        # Accept both ndarray and (nested) list formats
        if not isinstance(object_transform, np.ndarray):
            try:
                obj_mat = np.array(object_transform, dtype=float).reshape(4, 4)
            except Exception:
                raise TypeError("object_transform must be a 4x4 array-like structure")
        else:
            obj_mat = object_transform

        if obj_mat.shape != (4, 4):
            raise ValueError("object_transform must be 4x4 matrix")

        # Combine rotations: first apply object transform, then Y-axis rotation
        transform[:3, :3] = rot_matrix @ obj_mat[:3, :3]
    else:
        transform[:3, :3] = rot_matrix
        
    transform[:3, 3] = position_flipped
    
    # Convert to structured format
    return {
        "rows": 4,
        "cols": 4,
        "data": numpy_to_python(transform.T.flatten())
    }

def save_stk_scene_state(objects, room_vertices, door_location, output_dir, window_locations=None, filename="stk_scene_state.json"):
    """
    Save the current scene state to a JSON file using SSTK-compatible format.
    
    Args:
        objects: List of (id, position, rotation, transform_matrix) tuples
        room_vertices: List of (x, y) room corner coordinates
        door_location: (x, y) position of the door or Cutout object
        output_dir: Directory to save the output file
        window_locations: List of (x, y) positions of windows or list of Cutout objects
        filename: Output filename
    """
    scene_state = {
        "format": "sceneState",
        "scene": {
            "version": "scene@1.0.2",
            "up": {"x": 0, "y": 1, "z": 0},
            "front": {"x": 0, "y": 0, "z": 1},
            "unit": 1.0,
            "assetSource": ["fpModel"],
            "object": []
        }
    }
    
    try:
        # Convert objects to SSTK format
        object_list = []
        for idx, obj in enumerate(objects):
            id, position, rotation, transform_matrix = obj
            
            # Create transform matrix
            transform = create_transform_matrix(position, rotation, transform_matrix)
            
            object_list.append({
                "id": str(idx),
                "modelId": "fpModel." + id.split("/")[-1],
                "index": idx,
                "parentIndex": -1,
                "transform": transform,
            })

        arch = create_stk_arch(room_vertices, door_location, window_locations)
        scene_state["scene"]["arch"] = arch
        scene_state["scene"]["object"] = object_list

        # path = output_dir / filename
        # with open(str(path), 'w') as f:
            # json.dump(scene_state, f, indent=4)
        # Scene state saving can be enabled if needed
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Error logging can be enabled if needed

    #########################################################
    # match coordinate system for scene eval
    # also save the scene state with the original coordinates (without HSSD metadata transform)
    try:
        object_list = []
        for idx, obj in enumerate(objects):
            id, position, rotation, transform_matrix = obj
            
            # Create transform matrix, do not pass object_transform to avoid double rotation in sceneeval
            transform = create_transform_matrix(position, rotation, None)
            
            object_list.append({
                "id": str(idx),
                "modelId": "fpModel." + id.split("/")[-1],
                "index": idx,
                "parentIndex": -1,
                "transform": transform,
            })

        arch = create_stk_arch(room_vertices, door_location, window_locations)
        scene_state["scene"]["arch"] = arch
        scene_state["scene"]["object"] = object_list

        # path = output_dir / filename.replace(".json", "_fixed.json")
        path = output_dir / filename
        with open(str(path), 'w') as f:
            json.dump(sv_fix_coordinates(scene_state), f, indent=4)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger = get_logger('hsm_core.utils.stk_utils')
        logger.error(f"Error saving scene state to {path}: {e}")

def create_stk_arch(room_vertices, door_location, window_locations=None):
    """
    Convert room vertices, door location, and window locations into SSTK architecture format.
    
    Args:
        room_vertices: numpy array of room corner points
        door_location: tuple/array of door position or Cutout object
        window_locations: list of tuple/array of window positions or list of Cutout objects
        
    Returns:
        Dictionary containing SSTK-compatible architecture data
    """

    WALL_DEPTH = 0.1
    arch_data = {
        "version": "arch@1.0.2",
        "up": {"x": 0, "y": 1, "z": 0},
        "front": {"x": 0, "y": 0, "z": 1},
        "scaleToMeters": 1.0,
        "defaults": {
            "Wall": {"depth": WALL_DEPTH, "extraHeight": 0.035},
            "Ceiling": {"depth": 0.05},
            "Floor": {"depth": 0.05},
            "Ground": {"depth": 0.08}
        },
        "elements": []
    }

    # Check if door_location is a Cutout object or a tuple
    door_cutout = None
    if isinstance(door_location, Cutout):
        door_cutout = door_location
        door_location = door_cutout.location
    
    # Check if window_locations contains Cutout objects or tuples
    window_cutouts = []
    if window_locations:
        if all(isinstance(w, Cutout) for w in window_locations):
            window_cutouts = window_locations
            window_locations = [w.location for w in window_cutouts]

    # Flip door location X-coordinate to match SSTK coordinate system
    door_point = [-float(door_location[0]), 0, float(door_location[1])] if door_location else None
    
    # Flip window locations X-coordinates to match SSTK coordinate system
    window_points = []
    if window_locations:
        for window_loc in window_locations:
            window_points.append([-float(window_loc[0]), 0, float(window_loc[1])])

    # Create walls from vertices
    for i in range(len(room_vertices)):
        start_point = room_vertices[i]
        end_point = room_vertices[(i + 1) % len(room_vertices)]
        
        # Convert points to SSTK format [x, 0, z] with flipped X
        start = [float(-start_point[0]), 0, float(start_point[1])]
        end = [float(-end_point[0]), 0, float(end_point[1])]
        
        wall = {
            "id": f"wall_{i}",
            "type": "Wall",
            "height": WALL_HEIGHT,
            "points": [start, end],
            "holes": [],
            "materials": [
                {
                    "name": "inside",
                    "texture": "wallp_0",
                    "diffuse": "#ffffff"
                },
                {
                    "name": "outside",
                    "texture": "bricks_1",
                    "diffuse": "#ffffff"
                }
            ],
            "depth": WALL_DEPTH,
            "roomId": f"0"
        }
        
        # Add door hole if this wall segment contains the door
        if door_cutout and door_cutout.closest_wall_index == i:
            # Use the projection_on_wall value directly from the Cutout object
            dist_from_start = door_cutout.projection_on_wall
            wall["holes"].append({
                "id": "door_0",
                "type": "Door",
                "box": {
                    "min": [dist_from_start - door_cutout.width / 2, 0],
                    "max": [dist_from_start + door_cutout.width / 2, door_cutout.height]
                }
            })
            # Door addition debug logging can be enabled if needed
        elif door_point is not None and not door_cutout:
            # Fallback to old method if door_cutout is not provided
            if is_point_on_line_segment(door_point, start, end, tolerance=0.2):
                dist_from_start = distance_along_wall(door_point, start, end)
                wall["holes"].append({
                    "id": "door_0",
                    "type": "Door",
                    "box": {
                        "min": [dist_from_start - DOOR_WIDTH / 2, 0],
                        "max": [dist_from_start + DOOR_WIDTH / 2, DOOR_HEIGHT]
                    }
                })
                # Door addition debug logging can be enabled if needed
        
        # Add window holes if this wall segment contains any windows
        window_count = 0
        
        # First try to use window_cutouts if available
        if window_cutouts:
            for idx, window in enumerate(window_cutouts):
                if window.closest_wall_index == i:
                    # Use the projection_on_wall value directly from the Cutout object
                    dist_from_start = window.projection_on_wall
                    wall["holes"].append({
                        "id": f"window_{window_count}",
                        "type": "Window",
                        "box": {
                            "min": [dist_from_start - window.width / 2, window.bottom_height],
                            "max": [dist_from_start + window.width / 2, window.bottom_height + window.height]
                        }
                    })
                    window_count += 1
                    # Window addition debug logging can be enabled if needed
        elif window_points:
            # Fallback to old method if window_cutouts are not provided
            for idx, window_point in enumerate(window_points):
                if is_point_on_line_segment(window_point, start, end, tolerance=0.2):
                    dist_from_start = distance_along_wall(window_point, start, end)
                    half_width = WINDOW_WIDTH / 2
                    
                    wall["holes"].append({
                        "id": f"window_{window_count}",
                        "type": "Window",
                        "box": {
                            "min": [dist_from_start - half_width, WINDOW_BOTTOM_HEIGHT],
                            "max": [dist_from_start + half_width, WINDOW_BOTTOM_HEIGHT + WINDOW_HEIGHT]
                        }
                    })
                    window_count += 1
                    logger.debug(f"Added window {window_count} to wall {i} at distance {dist_from_start} (fallback method)")
        
        arch_data["elements"].append(wall)

    # Add floor (using room vertices with flipped X)
    floor_points = [[float(-v[0]), 0, float(v[1])] for v in room_vertices]
    floor = {
        "id": "floor_0",
        "type": "Floor",
        "points": floor_points, 
        "materials": [{
            "name": "surface",
            "texture": "default_floor",
            "diffuse": "#ffffff"
        }],
        "depth": arch_data["defaults"]["Floor"]["depth"],
        "roomId": "0"
    }
    arch_data["elements"].append(floor)

    # # Add ceiling (same as floor but elevated)
    # ceiling = {
    #     "id": "ceiling_0",
    #     "type": "Ceiling",
    #     "points": [floor_points],  # Same points as floor
    #     "materials": [{
    #         "name": "surface",
    #         "texture": "default_ceiling",
    #         "diffuse": "#ffffff"
    #     }],
    #     "depth": arch_data["defaults"]["Ceiling"]["depth"]
    # }
    # arch_data["elements"].append(ceiling)
    
    return arch_data

def is_point_on_line_segment(point, start, end, tolerance=0.2):
    """
    Check if point lies on line segment within tolerance.
    
    Args:
        point: [x, y, z] point to check
        start: [x, y, z] start of line segment
        end: [x, y, z] end of line segment
        tolerance: Maximum distance from line segment
        
    Returns:
        bool: True if point is on line segment within tolerance
    """
    # Extract x and z coordinates (ignore y)
    point_np = np.array([point[0], point[2]])
    start_np = np.array([start[0], start[2]])
    end_np = np.array([end[0], end[2]])
    
    # Vector from start to end
    line_vec = end_np - start_np
    line_length = np.linalg.norm(line_vec)
    
    if line_length < 1e-6:  # Very short line segment
        return np.linalg.norm(point_np - start_np) <= tolerance
    
    # Normalize line vector
    line_unit_vec = line_vec / line_length
    
    # Vector from start to point
    point_vec = point_np - start_np
    
    # Project point vector onto line
    projection = np.dot(point_vec, line_unit_vec)
    
    # Check if projection is within line segment bounds (with tolerance)
    if projection < -tolerance or projection > line_length + tolerance:
        return False
    
    # Calculate perpendicular distance from point to line
    projected_point = start_np + projection * line_unit_vec
    perp_dist = np.linalg.norm(point_np - projected_point)
    
    return perp_dist <= tolerance

def distance_along_wall(point, start, end):
    """
    Calculate distance of point from start of wall along the wall direction.
    
    Args:
        point: [x, y, z] point on wall
        start: [x, y, z] start of wall
        end: [x, y, z] end of wall
        
    Returns:
        float: Distance along wall from start point
    """
    # Extract x and z coordinates (ignore y)
    point_np = np.array([point[0], point[2]])
    start_np = np.array([start[0], start[2]])
    end_np = np.array([end[0], end[2]])
    
    # Vector from start to end
    wall_vec = end_np - start_np
    wall_length = np.linalg.norm(wall_vec)
    
    if wall_length < 1e-6:  # Very short wall
        return 0.0
    
    # Normalize wall vector
    wall_unit_vec = wall_vec / wall_length
    
    # Vector from start to point
    point_vec = point_np - start_np
    
    # Project point vector onto wall direction
    projection = np.dot(point_vec, wall_unit_vec)
    
    # Ensure projection is within wall bounds
    projection = max(0.0, min(wall_length, projection))
    
    return projection