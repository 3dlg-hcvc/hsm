import json
import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon
from pathlib import Path
from omegaconf import DictConfig

from hsm_core.vlm.gpt import Session, extract_json
from hsm_core.retrieval.utils.transform_tracker import TransformInfo
from hsm_core.scene.manager import Scene
from hsm_core.scene.objects import SceneObject
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.grid_utils import calculate_door_angle
from hsm_core.scene.scene_3d import create_scene_objects_from_motif
from hsm_core.solvers.solver_dfs import run_solver
from hsm_core.scene.motif import SceneMotif
from hsm_core.solvers import SceneSpatialOptimizer

GLOBAL_OFFSET = 0 # Offset for all objects to be closer to the wall
WALL_Z_OFFSET = 0.0001 # Offset for wall object to be closer to the wall
CEILING_Y_OFFSET = GLOBAL_OFFSET # Offset for ceiling object to be closer to the ceiling

def prepare_dfs_compatible_format(wall_data: dict, room_height: float) -> tuple[list, Polygon | None]:
    """
    Create a DFS solver compatible format for blockers on a specific wall.
    Processes blockers defined in wall_data["blocked_areas"].

    Args:
        wall_data (dict): Data dictionary for a single wall from extract_wall_data function.
                          Must contain 'start', 'end', 'length', 'angle', and 'blocked_areas'.
        room_height (float): The height of the room.

    Returns:
        tuple[list, Polygon | None]: Tuple containing:
            - List of fixed blocker objects in DFS solver compatible format.
            - Shapely Polygon representing the wall surface (length x height), or None if wall data is incomplete.
    """
    dfs_blocker_objects = []
    wall_id = wall_data.get("id", "unknown_wall")
    print(f"Preparing DFS compatible blockers for wall {wall_id}")

    # Extract necessary wall geometry data
    wall_start = wall_data.get("start")
    wall_end = wall_data.get("end")
    wall_length = wall_data.get("length")
    wall_angle = wall_data.get("angle")

    # Validate wall data required for processing
    wall_polygon = None
    if not all([wall_start, wall_end, wall_length is not None, wall_angle is not None]) or (wall_length is not None and wall_length <= 0):
        print(f"Warning: Skipping blocker generation and polygon creation for wall {wall_id} due to missing or invalid geometry data.")
        return [], None  # Return empty list and None for polygon

    # Ensure wall_length and room_height are floats for Polygon creation
    w_length = float(wall_length) if wall_length is not None else 0.0
    r_height = float(room_height)

    wall_polygon = Polygon([
        (0.0, 0.0),             # Bottom-left corner
        (w_length, 0.0),     # Bottom-right corner
        (w_length, r_height), # Top-right corner
        (0.0, r_height)      # Top-left corner
    ])
    print(f"  Created wall surface polygon for {wall_id}: Length={w_length:.2f}m, Height={r_height:.2f}m")

    # Process all defined blocked areas from wall_data
    blocked_areas = wall_data.get("processed_blocks", wall_data.get("blocked_areas", []))
    if not isinstance(blocked_areas, list):
        print(f"Warning: 'blocked_areas'/'processed_blocks' for wall {wall_id} is not a list. No blockers generated.")
    else:
        print(f"  Processing {len(blocked_areas)} blocked areas from wall data...")
        for block_idx, block in enumerate(blocked_areas):
            if not isinstance(block, dict):
                print(f"Warning: Item {block_idx} in blocked areas for wall {wall_id} is not a dictionary. Skipping.")
                continue

            block_start_pos = block.get("start")
            block_end_pos = block.get("end")
            block_height = block.get("height", room_height) # Use block height or full room height
            object_id = block.get("object_id", f"blocker_{block_idx}")
            block_type = "door" if block.get("is_door") else \
                         "window" if block.get("is_window") else \
                         "gap" if block.get("is_gap") else \
                         "furniture" if not block.get("is_wall_object") else \
                         "wall_obj"

            # Check if necessary block data exists and is numeric
            if not isinstance(block_start_pos, (int, float)) or not isinstance(block_end_pos, (int, float)):
                print(f"Warning: Invalid start/end position for {object_id} on wall {wall_id}. Skipping blocker.")
                continue

            # --- Calculate position and dimensions relative to wall ---
            # X-position: Center along the wall's length
            center_x_on_wall = (block_start_pos + block_end_pos) / 2.0
            
            # Base height (Y=0) or mount height (e.g., for paintings) or sill height (for windows) or wall height (for wall objects)
            base_y_on_wall = 0.0
            if block.get("is_window"):
                base_y_on_wall = block.get("sill_height", 0.9) # Windows start at sill height
            elif block.get("is_wall_object"):
                base_y_on_wall = block.get("mount_height", 0.0) # Default to floor if not specified
            
            # Y-position: Vertical center on the wall, relative to floor/base
            center_y_on_wall = base_y_on_wall + block_height / 2.0
            
            dim_along_wall = abs(block_end_pos - block_start_pos)
            dim_height_on_wall = block_height
            dim_depth_off_wall: float = 0.1

            # --- Create the fixed blocker object for DFS using wall coordinates ---
            dfs_obj = {
                "id": f"blocker_{wall_id}_{object_id}",
                "name": f"blocker_{block_type}_{wall_id}_{block_idx}",
                "dimensions": [dim_along_wall, dim_depth_off_wall, dim_height_on_wall], 
                "position": [center_x_on_wall, center_y_on_wall],
                "rotation": 0,
                "is_fixed": True,
                "ignore_collision": False,
            }
            dfs_blocker_objects.append(dfs_obj)
            print(f"    Added blocker: {dfs_obj['name']} at (X:{center_x_on_wall:.2f}, Y:{center_y_on_wall:.2f}) Dims=({dim_along_wall:.2f}W x {dim_height_on_wall:.2f}H)")

    return dfs_blocker_objects, wall_polygon


def call_llm_validated_json(
    session: Session,
    prompt_key: str,
    params: Dict[str, Any],
    validation_fn: Callable[[Any], Tuple[bool, str, int]],
    **kwargs,
) -> Dict[str, Any]:
    """Calls the VLM, validates the response, extracts and parses JSON."""
    print(f"Calling VLM prompt: {prompt_key}")
    response = session.send_with_validation(
        prompt_key, params, validation_fn, json=True, **kwargs
    )
    try:
        extracted_json_str = extract_json(response)
        if not extracted_json_str:
            raise ValueError("VLM response did not contain valid JSON.")
        return json.loads(extracted_json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from VLM response: {e}")
        print(f"Raw response: {response}")
        raise ValueError("Failed to parse JSON from VLM response.") from e
    except Exception as e:
        print(f"Error processing VLM response for {prompt_key}: {e}")
        raise


def calculate_world_pos_from_wall_params(
    item_params: Dict[str, Any],
    wall_data: Dict[str, Any],
    object_extents: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], float]:
    """
    Calculate 3D world position and rotation from wall-relative parameters.
    
    Args:
        item_params: Dictionary containing wall-relative parameters:
            - wall_position: Position along wall (0.0 to 1.0)
            - height_position: Height from floor
            - wall_distance: Distance from wall surface (optional, defaults to half object depth)
        wall_data: Dictionary containing wall data:
            - start: (x,z) start point of wall
            - end: (x,z) end point of wall
            - angle: Wall angle in degrees
            - length: Wall length
        object_extents: Tuple of (width, height, depth) of the object
        
    Returns:
        Tuple of (position, rotation) where:
        - position: (x,y,z) world coordinates
        - rotation: Rotation in degrees
    """
    try:
        # Extract wall parameters
        wall_start = np.array(wall_data["start"])
        wall_end = np.array(wall_data["end"])
        wall_angle = wall_data["angle"]
        wall_length = wall_data["length"]
        
        # Extract object parameters
        wall_pos_param = item_params.get("wall_position", 0.5) # Position along wall (0.0 to 1.0)
        height_param = item_params.get("height_position", 1.5) - object_extents[1] / 2 # Height from floor
        # Use half of object's depth as default distance from wall
        dist_param = item_params.get("wall_distance", object_extents[2] / 2) # Distance from wall surface
        
        # Calculate position along wall
        wall_vector = wall_end - wall_start
        wall_unit_vector = wall_vector / wall_length
        pos_along_wall = wall_start + wall_unit_vector * (wall_pos_param * wall_length)
        
        # Add offset perpendicular to wall
        perp_angle = wall_angle + 90
        pos_x = pos_along_wall[0] + dist_param * math.cos(math.radians(perp_angle))
        pos_z = pos_along_wall[1] + dist_param * math.sin(math.radians(perp_angle))
        
        # Combine into final position
        pos = (pos_x, height_param, pos_z)
        rot = wall_angle + 180 # Align with wall
        
        return pos, rot
        
    except Exception as e:
        print(f"Error calculating world position from wall params: {e}")
        raise

def calculate_world_pos_from_wall_solver(
    solver_pos: List[float],
    wall_data_item: Dict[str, Any],
    object_extents: Tuple[float, float, float],
    room_bounds: Optional[Tuple[float, float, float, float]] = None,
    motif: Optional[SceneMotif] = None,
) -> Tuple[Tuple[float, float, float], float]:
    """Converts wall solver coordinates (pos_along_wall, [depth_off_wall], height) back to 3D world coords, ensuring the object's back face is flush with the wall's front face."""
    wall_start = wall_data_item["start"]
    wall_end = wall_data_item["end"]
    wall_angle = wall_data_item["angle"]
    wall_length = wall_data_item["length"]
    # wall_thickness = wall_data_item.get("thickness", 0.0)  # Default to 0.0m if not specified

    solver_x_on_wall = solver_pos[0]
    solver_y_height = solver_pos[1]
    print(f"solver_x_on_wall: {solver_x_on_wall}, solver_y_height: {solver_y_height}")

    if wall_length <= 0:
        print(f"Warning: Wall {wall_data_item.get('id', 'unknown')} has zero length. Using origin with height/depth.")
        world_x = solver_x_on_wall * math.cos(math.radians(90))
        world_z = solver_x_on_wall * math.sin(math.radians(90))
        return (world_x, solver_y_height, world_z), 180

    wall_vector = np.array(wall_end) - np.array(wall_start)
    wall_unit_vector = wall_vector / wall_length

    clamped_solver_x = max(0.0, min(solver_x_on_wall, wall_length))
    world_pos_along = np.array(wall_start) + wall_unit_vector * clamped_solver_x
    world_x_along = world_pos_along[0]
    world_z_along = world_pos_along[1]

    # Calculate the inward normal (pointing INTO the room)
    wall_dx = wall_end[0] - wall_start[0]
    wall_dz = wall_end[1] - wall_start[1]
    
    # Calculate potential inward normal (90° counter-clockwise rotation of wall vector)
    potential_normal_x = -wall_dz / wall_length  
    potential_normal_z = wall_dx / wall_length
    
    # Determine if this normal points into the room by checking against room center
    wall_center_x = (wall_start[0] + wall_end[0]) / 2
    wall_center_z = (wall_start[1] + wall_end[1]) / 2
    
    # Calculate room center from bounds or use default fallback
    if room_bounds is not None:
        min_x, min_z, max_x, max_z = room_bounds
        room_center_x = (min_x + max_x) / 2
        room_center_z = (min_z + max_z) / 2
    else:
        raise ValueError("No room bounds provided")
    
    # Vector from wall center to room center
    to_room_center_x = room_center_x - wall_center_x
    to_room_center_z = room_center_z - wall_center_z
    
    # Check if our potential normal points toward room center
    dot_product = potential_normal_x * to_room_center_x + potential_normal_z * to_room_center_z
    
    # If dot product is positive, normal points toward room center (correct)
    # If negative, flip the normal
    if dot_product >= 0:
        inward_normal_x = potential_normal_x
        inward_normal_z = potential_normal_z
    else:
        inward_normal_x = -potential_normal_x
        inward_normal_z = -potential_normal_z
    
    # Handle HSSD normalization
    hssd_transform_info: Optional[TransformInfo] = None
    if motif is not None:
        # Check for HSSD transform using multiple methods for robustness
        if hasattr(motif, 'arrangement') and motif.arrangement and hasattr(motif.arrangement, 'objs'):
            for obj in motif.arrangement.objs:
                # Method 1: Use the transform tracker to check for HSSD alignment
                if obj.has_hssd_alignment():
                    hssd_transform_info = obj.get_hssd_alignment_transform()
                    if hssd_transform_info is not None:
                        print(f"Detected HSSD transform for wall object {motif.id} using transform tracker")
                        print(f"obj.transform: {obj.get_hssd_alignment_transform()}")
                    break

    # This correctly handles all cases: single objects (HSSD or not) and composite motifs.
    object_depth = 0.0
    if object_extents is not None and len(object_extents) > 2:
        object_depth = object_extents[2]

    offset = (object_depth/2) + WALL_Z_OFFSET
    
    # Compute tentative world position (object center)
    world_x = world_x_along + inward_normal_x * offset
    world_z = world_z_along + inward_normal_z * offset

    # ---------------------------------------------------------------------
    # Robustness: Ensure the computed position is inside the room bounds. If
    # it is not (e.g., negative Z due to an incorrect normal choice for
    # south-facing walls), flip the normal direction which effectively moves
    # the object towards the opposite side of the wall.
    # ---------------------------------------------------------------------
    if room_bounds is not None:
        min_x, min_z, max_x, max_z = room_bounds
        out_of_bounds = (
            world_x < min_x - 1e-3 or world_x > max_x + 1e-3 or
            world_z < min_z - 1e-3 or world_z > max_z + 1e-3
        )

        if out_of_bounds:
            # Flip direction
            world_x = world_x_along - inward_normal_x * offset
            world_z = world_z_along - inward_normal_z * offset

    pos = (world_x, solver_y_height, world_z)

    # Object rotation: face OUTWARD from the wall (aligned with wall surface)
    # For wall objects, use the wall angle to determine proper orientation
    rot = (wall_angle + 180) % 360  # Face outward from wall

    return pos, rot


def convert_world_to_wall_solver_coords(
    motif: SceneMotif,
    wall_data_item: Dict[str, Any],
) -> Optional[Tuple[List[float], float]]:
    """Converts 3D world coordinates of a motif to 2D wall surface coordinates for the solver."""
    if not hasattr(motif, "position") or motif.position is None:
        print(f"Warning: Motif {motif.id} has no position. Cannot convert to wall coordinates.")
        return None

    wall_start = wall_data_item.get("start")
    wall_end = wall_data_item.get("end")
    wall_length = wall_data_item.get("length")

    if not wall_start or not wall_end or wall_length is None:
        print(f"Warning: Missing wall geometry data for {wall_data_item.get('id', 'unknown')}. Cannot convert coords for {motif.id}")
        return None
    if wall_length <= 0:
        print(f"Warning: Wall {wall_data_item.get('id', 'unknown')} has zero length. Cannot convert coords for {motif.id}")
        return None # Or return [0, height]?

    # Calculate position relative to wall start
    motif_world_pos = np.array(motif.position)
    wall_start_vec = np.array(wall_start)
    wall_end_vec = np.array(wall_end)
    wall_vector = wall_end_vec - wall_start_vec
    wall_unit_vector = wall_vector / wall_length

    object_vec_world_xz = np.array([motif_world_pos[0], motif_world_pos[2]])
    object_vec_relative_to_start = object_vec_world_xz - wall_start_vec
    pos_along_wall = np.dot(object_vec_relative_to_start, wall_unit_vector)

    # Position for solver: [distance_along_wall, height_from_floor]
    solver_pos_x = pos_along_wall
    solver_pos_y = motif.position[1] # Height from floor is the Y coord

    # Rotation relative to wall surface is assumed 0 for solver
    solver_rotation = 0

    return [solver_pos_x, solver_pos_y], solver_rotation


def _prepare_base_solver_object(motif: SceneMotif, is_fixed: bool = False) -> Dict[str, Any]:
    """Creates a base dictionary for a solver object from a motif."""
    pos = [0.0, 0.0] # Default 2D position
    if hasattr(motif, 'position') and motif.position is not None:
         # Solver typically uses XZ plane from world coords
        pos = [motif.position[0], motif.position[2]]

    dims = [1.0, 1.0, 1.0] # Default dimensions
    if hasattr(motif, 'extents') and motif.extents is not None and np.any(motif.extents):
        dims = list(motif.extents)

    rot = 0.0
    if hasattr(motif, 'rotation'):
        rot = motif.rotation

    ignore_collision = False
    if hasattr(motif, 'ignore_collision'):
        ignore_collision = motif.ignore_collision

    return {
        "id": motif.id,
        "name": motif.id,
        "dimensions": dims,
        "position": pos, # Placeholder, will be overwritten by specific preppers
        "rotation": rot, # Placeholder, will be overwritten by specific preppers
        "is_fixed": is_fixed,
        "ignore_collision": ignore_collision,
    }


def _update_solver_object_position_rotation(solver_obj: Dict[str, Any], motif: SceneMotif) -> None:
    """Update solver object position and rotation from motif."""
    if hasattr(motif, 'position') and motif.position:
        solver_obj["position"] = [motif.position[0], motif.position[2]]
    if hasattr(motif, 'rotation'):
        solver_obj["rotation"] = motif.rotation


def prepare_large_solver_inputs(
    motifs_to_place: List[SceneMotif],
    fixed_motifs: List[SceneMotif],
    scene: Scene,
) -> List[Dict[str, Any]]:
    """Prepares the list of objects for the large object solver."""
    solver_objects = []

    # Add door clearance object if door location exists
    if scene.door_location and scene.room_polygon:
        door_x, door_z = scene.door_location
        door_angle = calculate_door_angle(scene.door_location, scene.room_polygon)

        # Calculate door clearance position inside the room
        clearance_offset = 0.5
        door_normal = (math.cos(math.radians(door_angle)), math.sin(math.radians(door_angle)))
        clearance_x = door_x + door_normal[0] * clearance_offset
        clearance_z = door_z + door_normal[1] * clearance_offset

        door_clearance = {
            "id": "door_clearance",
            "name": "door_clearance",
            "dimensions": [1.0, 0.1, 1.0],  # Use non-zero height for solver WHD
            "position": [clearance_x, clearance_z], # Solver uses 2D XZ plane
            "rotation": door_angle,
            "is_fixed": True,
            "ignore_collision": False, # Door clearance should block
        }
        solver_objects.append(door_clearance)

    # Add motifs to be placed by the solver
    for motif in motifs_to_place:
        solver_obj = _prepare_base_solver_object(motif, is_fixed=False)
        # Large object solver uses world XZ coordinates directly
        _update_solver_object_position_rotation(solver_obj, motif)
        # Include wall alignment info if available from VLM positioning step
        if hasattr(motif, 'wall_alignment') and motif.wall_alignment:
            solver_obj["wall_alignment"] = True
            if hasattr(motif, 'wall_alignment_id') and motif.wall_alignment_id is not None:
                solver_obj["wall_alignment_id"] = motif.wall_alignment_id
                print(f"  - Transferring wall alignment to solver for {motif.id}: wall_id={motif.wall_alignment_id}")
        solver_objects.append(solver_obj)

    # Add existing motifs as fixed obstacles
    for motif in fixed_motifs:
        solver_obj = _prepare_base_solver_object(motif, is_fixed=True)
        # Ensure position uses world XZ
        _update_solver_object_position_rotation(solver_obj, motif)
        solver_objects.append(solver_obj)

    print(f"Prepared {len(solver_objects)} inputs for large object solver.")
    return solver_objects


def prepare_wall_solver_inputs(
    motifs_to_place: List[SceneMotif],
    wall_data_item: Dict[str, Any],
    room_height: float,
) -> Tuple[List[Dict[str, Any]], Optional[Polygon]]:
    """Prepares inputs for the wall solver (blockers + placeable objects in wall coords)."""
    solver_objects = []
    wall_id = wall_data_item.get("id", "unknown_wall")

    # 1. Get fixed blockers for this wall
    # Needs the prepare_dfs_compatible_format function
    dfs_wall_blockers, wall_polygon = prepare_dfs_compatible_format(wall_data_item, room_height)
    if wall_polygon is None:
        print(f"Warning: Could not create wall polygon for {wall_id}. Skipping wall solver.")
        return [], None
    solver_objects.extend(dfs_wall_blockers)

    # 2. Prepare the wall objects to be placed (non-fixed) in wall coordinates
    for motif in motifs_to_place:
        # Directly use the VLM's suggested position if available
        # Assuming the VLM position [x, y] is stored in motif.llm_suggested_wall_pos
        if hasattr(motif, 'llm_suggested_wall_pos') and getattr(motif, 'llm_suggested_wall_pos', None):
             solver_pos = getattr(motif, 'llm_suggested_wall_pos')
             # Use 0 for rotation unless specified otherwise by VLM/logic
             solver_rot = getattr(motif, 'llm_suggested_wall_rot', 0)
             print(f"Using direct VLM position for {motif.id}: {solver_pos}")
        else:
            # Fallback to existing conversion if direct VLM position isn't available
            print(f"Warning: Direct VLM position not found for {motif.id}. Falling back to coordinate conversion.")
            wall_coords_result = convert_world_to_wall_solver_coords(motif, wall_data_item)
            if wall_coords_result is None:
                print(f"Warning: Could not convert motif {motif.id} to wall coordinates. Skipping.")
                continue
            solver_pos, solver_rot = wall_coords_result

        # Dimensions for wall solver: (width_on_wall, depth_off_wall, height_on_wall)
        solver_dim_w = motif.extents[0] # Width along the wall
        solver_dim_h = motif.extents[1] # Height
        solver_dim_d = motif.extents[2] # Depth off the wall

        solver_obj = _prepare_base_solver_object(motif, is_fixed=False)
        solver_obj["dimensions"] = [solver_dim_w, solver_dim_d, solver_dim_h] # correct dimensions for solver

        
        # Assign the determined position (either direct VLM or converted)
        solver_obj["position"] = solver_pos # [pos_along_wall, height_from_floor]
        solver_obj["rotation"] = 0 # Force rotation to 0 for DFS solver (wall objects face outward by default)
        solver_obj["wall_id"] = wall_id # Store wall_id for update step
        
        # Add rotation constraint to ensure DFS solver keeps rotation at 0
        if "constraints" not in solver_obj:
            solver_obj["constraints"] = []
        solver_obj["constraints"].append({
            "type": "global",
            "constraint": "rotation", 
            "angle": 0.0
        })

        solver_objects.append(solver_obj)

    print(f"Prepared {len(solver_objects)} inputs ({len(dfs_wall_blockers)} blockers, {len(motifs_to_place)} placeable) for wall solver on {wall_id}.")
    return solver_objects, wall_polygon


def prepare_ceiling_solver_inputs(
    motifs_to_place: List[SceneMotif],
    fixed_motifs: List[SceneMotif] # Optional fixed ceiling elements
) -> List[Dict[str, Any]]:
    """Prepares the list of objects for the ceiling object solver."""
    solver_objects = []

    # Add motifs to be placed by the solver
    for motif in motifs_to_place:
        solver_obj = _prepare_base_solver_object(motif, is_fixed=False)
        # Ceiling solver likely uses world XZ coordinates
        _update_solver_object_position_rotation(solver_obj, motif)
        solver_objects.append(solver_obj)

    # Add existing fixed motifs
    for motif in fixed_motifs:
        solver_obj = _prepare_base_solver_object(motif, is_fixed=True)
        _update_solver_object_position_rotation(solver_obj, motif)
        solver_objects.append(solver_obj)

    print(f"Prepared {len(solver_objects)} inputs for ceiling object solver.")
    return solver_objects



UpdateMotifFunc = Callable[[Dict[str, Any], SceneMotif, Optional[Dict]], None]

def run_solver_and_update_motifs(
    solver_inputs: List[Dict[str, Any]],
    geometry: Polygon,
    target_motifs_list: List[SceneMotif], # The list containing motifs to be updated
    output_dir: str,
    subfix: str,
    enable_solver: bool,
    update_func: UpdateMotifFunc,
    update_context: Optional[Dict] = None, # Extra context for the update func (e.g., wall_data)
    solver_fallback: bool = True
) -> Tuple[List[Dict[str, Any]], float]:
    """Runs the solver and updates motif positions using a provided function."""

    if not enable_solver:
        print(f"Solver disabled for {subfix}. Using initial positions.")
        # Return original positions (or indicate no change) and 0 occupancy change.
        # The motifs in target_motifs_list retain their pre-solver positions.
        # We return the original solver_inputs structure but with potentially modified 'is_fixed' if logic demands.
        original_placed = [m for m in solver_inputs if not m.get('is_fixed', False)]
        return original_placed, 0.0

    if not solver_inputs:
        print(f"No inputs provided to solver for {subfix}. Skipping.")
        return [], 0.0

    print(f"Running solver for {subfix}...")
    try:
        solved_objects, occupancy = run_solver(
            solver_inputs, geometry, output_dir=output_dir, subfix=subfix, enable=enable_solver, fallback=solver_fallback
        )
    except Exception as e:
        print(f"Error running solver for {subfix}: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("Returning empty results, motifs will not be updated by solver.")
        return [], 0.0 # Return empty list and zero occupancy on solver error


    print(f"Solver finished for {subfix}. Updating {len(target_motifs_list)} target motifs based on {len(solved_objects)} solved objects.")
    updated_count = 0
    # Create a lookup for faster access if many motifs
    motif_lookup = {motif.id: motif for motif in target_motifs_list}

    for solved_item in solved_objects:
        motif_id = solved_item.get("id")
        # Skip fixed objects like blockers or door clearance
        if solved_item.get("is_fixed", False):
            # print(f"  Skipping fixed solver item: {motif_id}") # Debug
            continue

        # Ensure motif_id is a string for lookup
        if motif_id is None:
            print(f"Warning: Solved object has no ID. Skipping.")
            continue
        
        matching_motif = motif_lookup.get(str(motif_id))

        if matching_motif:
            try:
                update_func(solved_item, matching_motif, update_context)
                updated_count += 1
            except Exception as e:
                print(f"Error updating motif {motif_id} using provided update function: {e}")
                print(traceback.format_exc())
                # Continue to next motif
        else:
            # Skip known blockers that aren't in target motif list
            if not str(motif_id).startswith("blocker_"):
                print(f"Warning: Solved object ID '{motif_id}' not found in target motif list for {subfix}.")

    print(f"Successfully updated {updated_count} motifs from solver results for {subfix}.")
    return solved_objects, occupancy




def update_large_motif_from_solver(
    solved_item: Dict[str, Any], motif: SceneMotif, context: Optional[Dict] = None
) -> None:
    """Updates a large object motif's position and rotation from solver results."""
    pos_2d = solved_item.get("position")
    rotation = solved_item.get("rotation")
    ignore_collision = solved_item.get("ignore_collision", False) # Persist ignore flag

    if pos_2d and len(pos_2d) >= 2 and rotation is not None:
        pos = (pos_2d[0], 0, pos_2d[1]) # X, Y=0 (floor level), Z coordinates
        print(f"  Updating LARGE motif {motif.id}: Pos={pos}, Rot={rotation:.1f}")
        motif.position = pos
        motif.rotation = rotation
        motif.object_type = ObjectType.LARGE # Ensure type
        motif.ignore_collision = ignore_collision # Update ignore status
    else:
        print(f"Warning: Solver result for large motif {motif.id} missing position/rotation. Skipping update.")


def update_wall_motif_from_solver(
    solved_item: Dict[str, Any], motif: SceneMotif, context: Optional[Dict] = None
) -> None:
    """Updates a wall object motif's position and rotation using wall-specific conversion."""
    solver_pos_2d = solved_item.get("position") # Should be [x_along, depth_off, y_height]

    if not context or "wall_data" not in context:
         print(f"Warning: Missing 'wall_data' in context for updating wall motif {motif.id}. Skipping.")
         return

    # Explicitly check for None or use np.any for array-like extents
    if not hasattr(motif, 'extents') or motif.extents is None or not np.any(motif.extents):
        print(f"Warning: Motif {motif.id} missing valid extents. Cannot calculate depth fallback. Skipping update.")
        return

    wall_data_item = context["wall_data"]

    if solver_pos_2d:
        # The solver provides the 2D position on the wall surface.
        # solver_pos_2d[0] is position along wall length.
        # The solver's y-coordinate is the height of the object's CENTER from the floor.
        
        # Convert the solver's 2D wall position to a 3D world position.
        room_bounds = context.get("room_bounds") if context else None
        
        # This function returns the desired CENTER of the object in 3D space.
        pos, rot = calculate_world_pos_from_wall_solver(solver_pos_2d, wall_data_item, motif.extents, room_bounds, motif)

        # The `preprocess_object_mesh` function normalizes the mesh so its origin is at the BOTTOM-center.
        # Therefore, we must adjust the calculated center Y position to be the bottom Y position
        # before assigning it to the motif.
        center_y = pos[1]
        height = motif.extents[1]
        bottom_y = center_y - (height / 2)
        
        final_pos = (pos[0], bottom_y, pos[2])

        print(f"  Updating WALL motif {motif.id} on wall {wall_data_item.get('id', 'unknown')}: WorldPos={final_pos}, WorldRot={rot:.1f} (from solver pos {solver_pos_2d})")
        motif.position = final_pos
        motif.rotation = rot
        motif.object_type = ObjectType.WALL # Ensure type
    else:
        print(f"Warning: Solver result for wall motif {motif.id} missing position data. Skipping update.")


def update_ceiling_motif_from_solver(
    solved_item: Dict[str, Any], motif: SceneMotif, context: Optional[Dict] = None
) -> None:
    """Updates a ceiling object motif's position and rotation."""
    pos_2d = solved_item.get("position")
    rotation = solved_item.get("rotation")

    if not context or "room_height" not in context:
        print(f"Warning: Missing 'room_height' in context for updating ceiling motif {motif.id}. Skipping.")
        # Default height or skip? Skipping for now.
        return

    room_height = context["room_height"]

    if pos_2d and len(pos_2d) >= 2 and rotation is not None:
        # Position Y coordinate is fixed at room height with ceiling offset
        # Use CEILING_Y_OFFSET for consistency with main.py ceiling positioning
        pos = (pos_2d[0], room_height + CEILING_Y_OFFSET, pos_2d[1])
        print(f"  Updating CEILING motif {motif.id}: Pos={pos}, Rot={rotation:.1f}")
        motif.position = pos
        motif.rotation = rotation
        motif.object_type = ObjectType.CEILING # Ensure type
    else:
        print(f"Warning: Solver result for ceiling motif {motif.id} missing position/rotation. Skipping update.") 



def filter_motifs_needing_optimization(motifs: List[SceneMotif]) -> List[SceneMotif]:
    """
    Filter motifs to only include those that need spatial optimization.
    
    Args:
        motifs: List of motifs to filter
        
    Returns:
        List of motifs that need optimization (not already optimized)
    """
    return [
        motif for motif in motifs 
        if not getattr(motif, 'is_spatially_optimized', False)
    ]

def run_spatial_optimization_for_stage(
    scene: Scene,
    cfg: DictConfig,
    current_stage_motifs: List[SceneMotif],
    object_type: ObjectType,
    output_dir: Path,
    stage_name: str = ""
) -> None:
    """
    Run spatial optimization for motifs from a specific processing stage.
    
    This is the primary spatial optimization mechanism that provides hierarchical optimization
    during object placement stages (large → wall → ceiling → small). This approach is preferred
    over global optimization as it maintains proper object placement hierarchy and artistic intent.
    
    The function supports both motif-level and object-level optimization based on configuration.
    Scene objects are created once and cached to avoid duplication.
    
    Args:
        scene: Scene object containing all motifs
        cfg: Configuration object
        current_stage_motifs: List of motifs from the current processing stage that need optimization
        object_type: Type of objects being processed
        output_dir: Output directory for saving optimization stats
        stage_name: Optional name for the stage (for logging/debugging)
    """
    print(f"\n{'='*80}")
    print(f"Spatial optimization - {object_type.name} {stage_name}")
    print(f"{'='*80}")
    
    if not current_stage_motifs:
        print(f" No motifs to optimize for {object_type.name} {stage_name}")
        return
    
    print(f" Stage Overview:")
    print(f"   - Processing {len(current_stage_motifs)} motifs")
    print(f"   - Object type: {object_type.name}")
    print(f"   - Stage name: {stage_name}")
    print(f"   - Motif IDs: {[m.id for m in current_stage_motifs]}")
    

    def _get_scene_objects_from_motif(motif: SceneMotif):
        """Return a list of SceneObjects belonging to motif."""
        print(f" Getting scene objects for motif: {motif.id}")

        try:
            objects = create_scene_objects_from_motif(motif)
            motif.add_objects(objects)
            return objects
        except Exception as exc:
            print(f" Error creating scene objects for motif '{motif.id}': {exc}")
            traceback.print_exc()
            return []
    
    # Always create scene objects from motifs, even if spatial optimization is disabled
    try:
        current_stage_objects = []
        for i, motif in enumerate(current_stage_motifs):
            print(f"Processing motif {i+1}/{len(current_stage_motifs)}: {motif.id}")
            scene_objects = _get_scene_objects_from_motif(motif)
            current_stage_objects.extend(scene_objects)
        
        print(f"\n Scene Object Creation Summary:")
        print(f"   - Total objects created: {len(current_stage_objects)}")
        # print(f"   - From {len(current_stage_motifs)} motifs")
        # print(f"   - Object breakdown:")
        object_counts = {}
        for obj in current_stage_objects:
            obj_type_name = obj.obj_type.name
            object_counts[obj_type_name] = object_counts.get(obj_type_name, 0) + 1
        # for obj_type_name, count in object_counts.items():
            # print(f"     * {obj_type_name}: {count}")
        
        if not current_stage_objects:
            print(f" No scene objects created for {object_type.name} {stage_name}, skipping optimization")
            return

    except Exception as e:
        print(f" Error during scene object creation for {object_type.name} {stage_name}: {e}")
        print("Continuing without scene object creation")
        traceback.print_exc()
        return
    
    if not cfg.mode.get('enable_spatial_optimization', False):
        print(f"  Skip scene spatial optimization from config for {object_type.name} {stage_name}")
        return
    
    try:
        from hsm_core.solvers.unified_optimizer import SceneSpatialOptimizer
        from hsm_core.solvers.config import SceneSpatialOptimizerConfig

        # Configure spatial optimizer for this object type
        config = SceneSpatialOptimizerConfig()
        config.debug_output = True
        config.use_motif_level_optimization = cfg.mode.get('use_motif_level_optimization', True)

        context_motifs = [
            m for m in scene.scene_motifs 
            if m not in current_stage_motifs
        ]
        print(f"   - Found {len(context_motifs)} context motifs (was filtering by object_type)")
        for motif in context_motifs:
            obj_type_name = motif.object_type.name if hasattr(motif, 'object_type') else 'unknown'
            print(f"     * {motif.id} ({obj_type_name})")
        
        context_objects = []
        for motif in context_motifs:
            try:
                scene_objects = _get_scene_objects_from_motif(motif)
                context_objects.extend(scene_objects)
                print(f"     * {motif.id}: {len(scene_objects)} objects")
                
                # Log object names for debugging
                if scene_objects:
                    object_names = [obj.name for obj in scene_objects]
                    print(f"       Objects: {object_names}")
                    
            except Exception as e:
                print(f"       Error creating scene objects for context motif {motif.id}: {e}")
                # Continue with other motifs
        
        print(f"   - Total context objects: {len(context_objects)}")
        print(f"   - Context object breakdown:")
        context_counts = {}
        for obj in context_objects:
            obj_type_name = obj.obj_type.name
            context_counts[obj_type_name] = context_counts.get(obj_type_name, 0) + 1
        for obj_type_name, count in context_counts.items():
            print(f"     * {obj_type_name}: {count}")
        
        # Initialize unified spatial optimizer (cached on scene to preserve meshes across stages)
        print(f"\n  Initializing Optimizer:")
        print("-" * 40)
        
        # Check if we already have a cached optimizer on the scene
        if not hasattr(scene, '_cached_spatial_optimizer') or scene._cached_spatial_optimizer is None:
            print("   - Creating new optimizer (first time)")
            optimizer = SceneSpatialOptimizer(scene, config)
            scene._cached_spatial_optimizer = optimizer
            
            # Initialize room geometry once
            try:
                print("   - Initializing room geometry...")
                optimizer._initialize_room_geometry()
                print("    Room geometry initialized")
            except Exception as preload_e:
                print(f"     Failed to initialize room geometry: {preload_e}")
        else:
            print("   - Reusing cached optimizer (preserves mesh cache)")
            optimizer = scene._cached_spatial_optimizer
            # Update config for this stage
            optimizer.config = config
        
        # Load meshes for current stage objects + context (incremental loading)
        try:
            print("    Loaded object meshes...")
            all_objects = current_stage_objects + context_objects
            optimizer._load_object_meshes(all_objects)
            print(f"    Loaded meshes for {len(all_objects)} objects")
            print(f"    Total cached meshes: {len(optimizer._object_meshes)}")
        except Exception as preload_e:
            print(f"     Failed to load object meshes: {preload_e}")
        
        # Optimize only current stage objects using context for collision/support checking
        print(f"\n Running Optimization:")
        print("-" * 40)
        optimized_current_objects = _optimize_stage_objects_only(
            optimizer, current_stage_objects, context_objects, current_stage_motifs, 
            config.use_motif_level_optimization, object_type
        )
        
        print(f"\n Scene spatial optimization completed for {object_type.name} {stage_name}")
        

        try:
            if optimized_current_objects:
                print("  Persisting optimized objects into motifs...")
                motif_to_objs: Dict[str, List[SceneObject]] = {}
                for obj in optimized_current_objects:
                    motif_id = getattr(obj, 'motif_id', None)
                    if motif_id is None:
                        continue
                    if motif_id not in motif_to_objs:
                        motif_to_objs[motif_id] = []
                    motif_to_objs[motif_id].append(obj)

                updated_motifs = 0
                for motif in current_stage_motifs:
                    objs = motif_to_objs.get(motif.id)
                    if not objs:
                        continue
                    # Overwrite objects by name inside the motif
                    motif.add_objects(objs)

                    # Apply the single translation used for the whole motif directly
                    try:
                        delta_applied = False
                        for o in objs:
                            if getattr(o, 'optimized_world_pos', None) is not None:
                                dx = float(o.optimized_world_pos[0]) - float(o.position[0])
                                dy = float(o.optimized_world_pos[1]) - float(o.position[1])
                                dz = float(o.optimized_world_pos[2]) - float(o.position[2])
                                motif.position = (
                                    float(motif.position[0]) + dx,
                                    float(motif.position[1]) + dy,
                                    float(motif.position[2]) + dz,
                                )
                                delta_applied = True
                                break
                        if not delta_applied:
                            print(f"   Info: No optimized_world_pos found for motif {motif.id}; leaving position unchanged")
                    except Exception as pos_e:
                        print(f"   Warning: Failed to apply motif translation for {motif.id}: {pos_e}")

                    motif.is_spatially_optimized = True
                    updated_motifs += 1

                print(f"  Updated {updated_motifs} motif(s) with optimized objects")
        except Exception as persist_e:
            print(f"  Warning: Failed to persist optimized results into motifs: {persist_e}")
        
    except Exception as e:
        print(f" Error during spatial optimization for {object_type.name} {stage_name}: {e}")
        print("Continuing without spatial optimization")
        traceback.print_exc()


def _optimize_stage_objects_only(
    optimizer: 'SceneSpatialOptimizer',
    current_stage_objects: List,
    context_objects: List,
    current_stage_motifs: List[SceneMotif],
    use_motif_optimization: bool,
    object_type: ObjectType
) -> List:
    """
    Enhanced version of _optimize_stage_objects_only with comprehensive logging.
    
    This function provides detailed debugging information to track down why small objects
    are frequently reported as "not supported".
    """
    print(f"\n OPTIMIZATION DEBUG:")
    print(f"   - Current stage objects: {len(current_stage_objects)}")
    print(f"   - Context objects: {len(context_objects)}")
    print(f"   - Motifs: {len(current_stage_motifs)}")
    print(f"   - Use motif optimization: {use_motif_optimization}")
    print(f"   - Object type: {object_type.name}")
    
    if not current_stage_objects:
        print(" No objects to optimize")
        return []
    
    if use_motif_optimization:
        print(f"\n MOTIF-LEVEL OPTIMIZATION:")
        print("-" * 40)
        
        # Group current stage objects by motif for motif-level optimization
        motif_groups = {}
        for obj in current_stage_objects:
            motif_id = getattr(obj, 'motif_id', 'unknown')
            if motif_id not in motif_groups:
                motif_groups[motif_id] = []
            motif_groups[motif_id].append(obj)
        
        print(f"   Grouped into {len(motif_groups)} motifs:")
        for motif_id, objs in motif_groups.items():
            print(f"     - {motif_id}: {len(objs)} objects")
        
        optimized_objects = []
        
        # Optimize each motif as a unit against context + other current stage objects
        for i, (motif_id, motif_objects) in enumerate(motif_groups.items()):
            print(f"\n Optimizing motif {i+1}/{len(motif_groups)}: {motif_id}")
            print(f"   Objects in motif: {len(motif_objects)}")
            for obj in motif_objects:
                print(f"     - {obj.name} ({obj.obj_type.name}) at {obj.position}")
            
            # Use motif_id comparison rather than direct object identity so that the
            # check remains valid even after `current_stage_objects` entries are
            # replaced by their optimised copies in previous iterations.
            other_current_objects = [obj for obj in current_stage_objects if getattr(obj, 'motif_id', 'unknown') != motif_id]
            full_context = context_objects + other_current_objects
            
            print(f"   Full context: {len(full_context)} objects")
            print(f"     - Context objects: {len(context_objects)}")
            print(f"     - Other current objects: {len(other_current_objects)}")
            
            # Use the optimizer's motif optimization logic
            try:
                print(f"    Starting motif optimization...")
                optimized_motif = optimizer._optimize_motif_as_unit(motif_objects, full_context)
                optimized_objects.extend(optimized_motif)
                
                print(f"    Motif optimization completed")
                print(f"   Results:")
                for orig, new in zip(motif_objects, optimized_motif):
                    if orig.position != new.position:
                        print(f"      {orig.name}: {orig.position} → {new.position}")
                    else:
                        print(f"      {orig.name}: position unchanged")

                # Update the shared `current_stage_objects` list **in-place** so
                # that subsequent motif optimisations (or individual object
                # passes) operate on the newest world poses.  Without this,
                # later optimisations may collide/support-check against stale
                # coordinates and undo earlier fixes.
                print(f"    Updating shared object list...")
                updated_count = 0
                for orig, new in zip(motif_objects, optimized_motif):
                    try:
                        idx = current_stage_objects.index(orig)
                        current_stage_objects[idx] = new  # overwrite with updated copy
                        updated_count += 1
                    except ValueError:
                        # Should not happen, but append as fail-safe
                        current_stage_objects.append(new)
                
                print(f"    Updated {updated_count} objects in shared list")
                    
            except Exception as e:
                print(f"    Error optimizing motif {motif_id}: {e}")
                traceback.print_exc()
                # Fallback to original objects if optimization fails
                optimized_objects.extend(motif_objects)
    
    print(f"\n Spatial optimization completed: {len(optimized_objects)} objects processed")
    return optimized_objects

def run_spatial_optimization_for_all_types(
    scene: Scene,
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> None:
    """
    Run motif-level spatial optimization for all motif types (LARGE, WALL, CEILING, SMALL).

    Args:
        scene: The Scene object containing all motifs.
        cfg: The configuration object.
        output_dir: The output directory for saving optimization results.

    This function iterates over all motif types, filters motifs that need optimization,
    and runs spatial optimization for each type as needed.
    """
    from hsm_core.scene.processing_helpers import run_spatial_optimization_for_stage, filter_motifs_needing_optimization
    from hsm_core.scene.core.objecttype import ObjectType

    all_motifs = scene.get_all_motifs()
    motif_types = [ObjectType.LARGE, ObjectType.WALL, ObjectType.CEILING, ObjectType.SMALL]

    for obj_type in motif_types:
        type_motifs = [m for m in all_motifs if m.object_type == obj_type]
        if not type_motifs:
            print(f"No {obj_type.name} motifs found, skipping...")
            continue
        motifs_needing_optimization = filter_motifs_needing_optimization(type_motifs)
        if motifs_needing_optimization:
            print(f"Running spatial optimization for {len(motifs_needing_optimization)} {obj_type.name} motifs...")
            try:
                run_spatial_optimization_for_stage(
                    scene=scene,
                    cfg=cfg,
                    current_stage_motifs=motifs_needing_optimization,
                    object_type=obj_type,
                    output_dir=output_dir,
                    stage_name=f"export_only_{obj_type.name.lower()}"
                )
            except Exception as e:
                print(f"Warning: Spatial optimization failed for {obj_type.name} motifs: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"All {obj_type.name} motifs are already optimized, skipping...")
    print("Motif-level spatial optimization completed.")