"""
Wall Object Processing Module
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import traceback
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.core.motif import SceneMotif
from hsm_core.scene.core.manager import Scene
from hsm_core.scene.core.spec import ObjectSpec, SceneSpec
from hsm_core.scene.geometry.wall_analysis import extract_wall_data, visualize_walls_as_surfaces
from hsm_core.scene.visualization.visualization import SceneVisualizer
from hsm_core.scene.processing.generate_scene_motif import process_scene_motifs
from hsm_core.utils import get_logger
from hsm_core.scene.validation.validate import (
    validate_arrangement_smc,
    validate_wall_objects,
    validate_wall_position,
)

logger = get_logger('scene.processing.wall')
from hsm_core.scene.processing.processing_helpers import (
    call_llm_validated_json,
    calculate_world_pos_from_wall_params,
    prepare_wall_solver_inputs,
    run_solver_and_update_motifs,
    update_wall_motif_from_solver,
    run_spatial_optimization_for_stage,
    filter_motifs_needing_optimization,
)
from hsm_core.scene.ablation import create_individual_scene_motifs_with_analysis

from hsm_core.vlm.gpt import Session
from hsm_core.vlm.utils import round_nested_values
from hsm_core.retrieval.model.model_manager import ModelManager
from hsm_core.config import PROMPT_DIR, PROJECT_ROOT
from hsm_core.scene.io.export import save_scene

def _get_eligible_walls(
    all_wall_data: List[Dict],
    wall_occupancy: Dict[str, float],
    threshold_percent: float
) -> Dict[str, Dict]:
    """
    Determine eligible walls that have sufficient space and haven't reached target occupancy.

    Args:
        all_wall_data: List of wall data dictionaries
        wall_occupancy: Current occupancy percentages for each wall
        threshold_percent: Target occupancy percentage

    Returns:
        Dictionary mapping wall_id to wall info for eligible walls
    """
    eligible_walls = {}
    for i, wall_item in enumerate(all_wall_data):
        wall_id = wall_item.get("id", f"wall_{i}")
        available_percent = wall_item.get("available_percent", 0)
        current_occupancy = wall_occupancy.get(wall_id, 0.0)

        if available_percent > threshold_percent and current_occupancy < threshold_percent:
            wall_info = wall_item.copy()
            wall_info["id"] = wall_id
            eligible_walls[wall_id] = wall_info

    return eligible_walls


def _assign_wall_objects_to_walls(
    wall_session: Session,
    wall_objects_spec: List[ObjectSpec],
    eligible_walls: Dict[str, Dict],
    room_description: str,
    updated_plot: Any,
    stage_name: str
) -> List[ObjectSpec]:
    """
    Assign wall objects to eligible walls using LLM.

    Args:
        wall_session: VLM session for wall processing
        wall_objects_spec: List of wall object specifications to assign
        eligible_walls: Dictionary of eligible walls
        room_description: Description of the room
        updated_plot: Current scene plot for visualization
        stage_name: Name for logging (e.g., "initial", "extra")

    Returns:
        List of successfully assigned object specifications
    """
    logger.info(f"Assigning {len(wall_objects_spec)} {stage_name} wall objects")

    try:
        assign_params = {
            "ROOM_DESCRIPTION": room_description,
            "WALL_OBJECTS": [round_nested_values(obj.to_gpt_dict(), 2) for obj in wall_objects_spec],
            "ELIGIBLE_WALLS": list(eligible_walls.keys()),
        }
        assign_validation = lambda r: validate_wall_objects(r, [obj.id for obj in wall_objects_spec], list(eligible_walls.keys()))
        assign_images = [updated_plot] if updated_plot else []
        wall_assignment_data = call_llm_validated_json(
            wall_session, "assign_wall_objects_to_walls", assign_params,
            assign_validation, verbose=True, images=assign_images
        )
        assignments = {item["id"]: item["wall_id"] for item in wall_assignment_data.get("objects", []) if "id" in item and "wall_id" in item}

        # Get wall data list for validation
        all_wall_data = list(eligible_walls.values())
        return _validate_and_assign_wall_objects(wall_objects_spec, assignments, all_wall_data, stage_name)

    except Exception as e:
        logger.error(f"Error during {stage_name} wall assignment: {e}. Skipping {stage_name} objects")
        return []


def _add_motifs_and_visualize(
    scene: Scene,
    motifs: List[SceneMotif],
    visualizer: SceneVisualizer,
    vis_output_dir: Path,
    iteration: int,
    stage_name: str
) -> Any:
    """
    Add motifs to scene and create visualization.

    Args:
        scene: Scene object to add motifs to
        motifs: List of motifs to add
        visualizer: SceneVisualizer instance
        output_dir: Output directory path
        iteration: Current iteration number
        stage_name: Stage name for visualization filename

    Returns:
        Updated plot object
    """
    if motifs:
        logger.info(f"Adding {len(motifs)} processed {stage_name} wall motifs to the scene")
        scene.add_motifs(motifs)
        logger.debug(f"Visualizing scene with {stage_name} wall objects")
        updated_plot, _ = visualizer.visualize(str(vis_output_dir / f"wall_iteration_{iteration}_{stage_name}.png"))
        return updated_plot
    else:
        logger.warning(f"No {stage_name} wall motifs were processed successfully")
        return None


def _check_all_walls_complete(wall_occupancy: Dict[str, float], threshold_percent: float) -> bool:
    """
    Check if all walls have reached the target occupancy percentage.

    Args:
        wall_occupancy: Dictionary mapping wall_id to occupancy percentage
        threshold_percent: Target occupancy percentage

    Returns:
        True if all walls have reached target occupancy, False otherwise
    """
    return all(occupancy >= threshold_percent for occupancy in wall_occupancy.values())


def _update_wall_occupancy(
    scene: Scene,
    wall_occupancy: Dict[str, float],
    door_location,
    window_locations
) -> None:
    """
    Update wall occupancy percentages based on current scene motifs.

    Args:
        scene: Scene object containing current motifs
        wall_occupancy: Dictionary to update with occupancy percentages
        door_location: Door location for wall data extraction
        window_locations: Window locations for wall data extraction
    """
    current_wall_data = extract_wall_data(
        room_polygon=scene.room_polygon,
        scene_motifs=scene.scene_motifs,
        door_location=door_location,
        window_locations=window_locations,
    )

    for wall in current_wall_data:
        wall_id = wall.get("id", f"wall_{current_wall_data.index(wall)}")
        available_percent = wall.get("available_percent", 100)
        wall_occupancy[wall_id] = 100 - available_percent


async def _process_assigned_wall_objects(
    wall_session: Session,
    wall_id: str,
    current_wall_data: Dict,
    wall_specific_specs: List[ObjectSpec],
    cfg: DictConfig,
    output_dir: Path,
    vis_output_dir: Path,
    room_description: str,
    model,
    scene_height: float,
    stage_suffix: str,
    scene: Scene,
    room_bounds: Optional[Tuple[float, float, float, float]] = None,
    solver_fallback: bool = True,
) -> Tuple[List[SceneMotif], float]:
    """
    Processes a list of object specs assigned to a specific wall.

    Args:
        wall_session: VLM session for wall processing
        wall_id: ID of the wall being processed
        current_wall_data: Wall geometry and availability data
        wall_specific_specs: List of object specs assigned to this wall
        cfg: Configuration object
        output_dir: Output directory path
        vis_output_dir: Output directory path for visualization
        room_description: Description of the room
        model: ModelManager instance
        scene_height: Height of the room
        stage_suffix: Suffix for file naming (e.g., "_initial", "_extra")
        scene: Scene instance
        room_bounds: Room bounds for positioning
        solver_fallback: Whether to use solver fallback

    Returns:
        Tuple of (processed_motifs, solver_occupancy) where:
        - processed_motifs: List of processed SceneMotif objects
        - solver_occupancy: Occupancy ratio (0-1) from the solver
    """
    logger.info(f"Processing wall {wall_id} ({len(wall_specific_specs)} objects) - Stage: {stage_suffix.strip('_')}")
    processed_motifs_for_wall: List[SceneMotif] = []
    current_wall_motifs: List[SceneMotif] = []
    current_wall_motifs_solved: List[SceneMotif] = []
    solver_occupancy = 0.0

    try:
        wall_motif_spec = {}
        arrangement_params = {"ROOM_TYPE": room_description, "WALL_OBJECTS": [obj.to_gpt_dict() for obj in wall_specific_specs]}
        wall_motif_spec = call_llm_validated_json(
            wall_session,"populate_surface_motifs", arrangement_params,
            lambda r: validate_arrangement_smc(r, [obj.id for obj in wall_specific_specs], wall_specific_specs), verbose=True,
        )
        
        if not cfg.mode.use_scene_motifs:
            analysis_str = create_individual_scene_motifs_with_analysis([obj.to_dict() for obj in wall_specific_specs], wall_motif_spec)
            validate_arrangement_smc(analysis_str, [obj.id for obj in wall_specific_specs], wall_specific_specs)
            wall_motif_spec = json.loads(analysis_str)

        # Process Motifs
        wall_motifs_processed, _ = await process_scene_motifs(
            wall_specific_specs, wall_motif_spec, output_dir, room_description, model,
            object_type=ObjectType.WALL,
        )
        valid_wall_motifs = [m for m in wall_motifs_processed if m.object_specs and getattr(m.object_specs[0], 'wall_id', None) == wall_id]

        if not valid_wall_motifs:
            logger.info(f"No valid motifs created for wall {wall_id} in stage {stage_suffix}")
            return [], 0.0 # Return empty list and zero occupancy if no motifs
        current_wall_motifs.extend(valid_wall_motifs)
        logger.info(f"Processed {len(current_wall_motifs)} motifs for wall {wall_id} stage {stage_suffix}")

        # Detailed Placement (VLM)
        wall_viz_path = str(vis_output_dir / f"wall_{wall_id}{stage_suffix}_surface.png")
        wall_viz_fig = visualize_walls_as_surfaces([current_wall_data], scene_height, add_grid_markers=True)
        if wall_viz_fig: plt.close(wall_viz_fig)

        # Apply VLM positioning
        placement_images = [wall_viz_path] if wall_viz_path and os.path.exists(wall_viz_path) else []
        position_params = {"WALL_OBJECTS": [m.to_gpt_dict() for m in current_wall_motifs], "ROOM_DESCRIPTION": room_description, "WALL_DATA": current_wall_data}
        detailed_position_data = call_llm_validated_json(
             wall_session, "position_wall_objects", position_params,
             lambda r: validate_wall_position(r, [m.id for m in current_wall_motifs]), verbose=True, images=placement_images,
         )

        temp_positioned_motifs = []
        positioned_count = 0
        for item in detailed_position_data.get("positions", []):
             motif_id = item.get("id")
             llm_pos_2d = item.get("position") # Get the [x, y] from VLM
             matching_motif = next((m for m in current_wall_motifs if m.id == motif_id), None)
             
             if matching_motif:
                 if llm_pos_2d and isinstance(llm_pos_2d, list) and len(llm_pos_2d) == 2:
                     setattr(matching_motif, 'llm_suggested_wall_pos', llm_pos_2d)

                 try:
                    pos, rot = calculate_world_pos_from_wall_params(item, current_wall_data, matching_motif.extents)
                    matching_motif.position = pos; matching_motif.rotation = rot; matching_motif.object_type = ObjectType.WALL
                    temp_positioned_motifs.append(matching_motif)
                    positioned_count += 1
                 except Exception as calc_e:
                    logger.error(f"Error calculating world pos for {motif_id} (wall {wall_id}, stage {stage_suffix}): {calc_e}")
             else:
                logger.warning(f"Motif {motif_id} from VLM position response not found (wall {wall_id}, stage {stage_suffix})")

        current_wall_motifs = temp_positioned_motifs # Keep only successfully positioned motifs
        if positioned_count == 0:
            logger.warning(f"VLM did not provide valid positions for any objects on wall {wall_id} stage {stage_suffix}. Skipping processing for this wall/stage")
            return [], 0.0
        logger.info(f"Initial positions calculated for {positioned_count} motifs on wall {wall_id} stage {stage_suffix}")

        # Collision Detection (Solver)
        if current_wall_motifs and cfg.mode.use_solver:
            solver_inputs, wall_polygon_geom = prepare_wall_solver_inputs(current_wall_motifs, current_wall_data, scene_height)
            if wall_polygon_geom and solver_inputs:
                update_context = {
                    "wall_data": current_wall_data,
                    "room_bounds": room_bounds
                }

                placed_motifs, solver_occupancy_result = run_solver_and_update_motifs(
                    solver_inputs, wall_polygon_geom, current_wall_motifs, str(vis_output_dir),
                    subfix=f"{wall_id}_collision", enable_solver=cfg.mode.use_solver,
                    update_func=update_wall_motif_from_solver, update_context=update_context,
                    solver_fallback=solver_fallback
                )
                solver_occupancy = solver_occupancy_result

                solved_ids = {m["id"] for m in placed_motifs} if placed_motifs else set()

                current_wall_motifs_solved = [m for m in current_wall_motifs if m.id in solved_ids]

                unsolved_motifs = [m for m in current_wall_motifs if m.id not in solved_ids]
                if unsolved_motifs:
                    logger.info(
                        f"Discarding {len(unsolved_motifs)} wall motif(s) that were not placed by the solver: "
                        f"{[m.id for m in unsolved_motifs]}"
                    )

                current_wall_motifs = current_wall_motifs_solved
            else:
                current_wall_motifs_solved = current_wall_motifs

        # Run spatial optimization after solver for current wall motifs
        motifs_needing_optimization = filter_motifs_needing_optimization(current_wall_motifs_solved)
        if motifs_needing_optimization:
            run_spatial_optimization_for_stage(
                scene=scene,
                cfg=cfg,
                current_stage_motifs=motifs_needing_optimization,
                object_type=ObjectType.WALL,
                output_dir=output_dir,
                stage_name=f"wall_{wall_id}{stage_suffix}"
            )

        if current_wall_motifs:
            processed_motifs_for_wall.extend(current_wall_motifs)
            logger.info(f"Successfully processed {len(processed_motifs_for_wall)} motifs for wall {wall_id} stage {stage_suffix}")

    except Exception as wall_proc_e:
         logger.error(f"Error processing wall {wall_id} stage {stage_suffix}: {wall_proc_e}")
         traceback.print_exc()
         return [], 0.0

    return processed_motifs_for_wall, solver_occupancy


def _validate_and_assign_wall_objects(
    wall_objects_spec: List[ObjectSpec],
    assignments: Dict[str, str],
    all_wall_data: List[Dict],
    stage_name: str = "wall objects"
) -> List[ObjectSpec]:
    """
    Helper function to validate and assign wall objects to walls.
    
    Args:
        wall_objects_spec: List of wall object specifications
        assignments: Dictionary mapping object ID to wall ID
        all_wall_data: List of all wall data dictionaries
        stage_name: Name for logging (e.g., "initial", "extra")
    
    Returns:
        List of successfully assigned object specifications
    """
    logger.debug(f"Updating {stage_name} object specifications with assigned wall IDs")
    all_wall_ids = {wall_item.get("id", f"wall_{i}") for i, wall_item in enumerate(all_wall_data)}

    assigned_specs = []
    for obj_spec in wall_objects_spec:
        assigned_wall_id = assignments.get(obj_spec.id)
        if assigned_wall_id and assigned_wall_id in all_wall_ids:
            if hasattr(obj_spec, 'wall_id'):
                setattr(obj_spec, 'wall_id', assigned_wall_id)
            assigned_specs.append(obj_spec)
            logger.debug(f"  - {obj_spec.id} assigned to {assigned_wall_id}")
        else:
            logger.warning(f"  - {obj_spec.id} was not assigned to a valid wall")
    
    return assigned_specs


async def _process_wall_objects_by_wall(
    assigned_specs: List[ObjectSpec],
    all_wall_data: List[Dict],
    wall_session: Session,
    cfg: DictConfig,
    output_dir_override: Path,
    vis_output_dir: Path,
    room_description: str,
    model,
    scene_height: float,
    stage_suffix: str,
    room_bounds: Optional[Tuple[float, float, float, float]],
    solver_fallback: bool,
    scene: Scene
) -> Tuple[List[SceneMotif], float]:
    """
    Helper function to process assigned wall objects by wall.

    Args:
        assigned_specs: List of assigned object specifications
        all_wall_data: List of all wall data dictionaries
        wall_session: VLM session for wall processing
        cfg: Configuration object
        output_dir_override: Output directory path
        room_description: Description of the room
        model: ModelManager instance
        scene_height: Height of the room
        stage_suffix: Suffix for file naming (e.g., "_initial", "_extra")
        room_bounds: Room bounds for positioning
        solver_fallback: Whether to use solver fallback
        scene: Scene instance

    Returns:
        Tuple of (processed_motifs, max_solver_occupancy) where:
        - processed_motifs: List of processed SceneMotif objects
        - max_solver_occupancy: Maximum solver occupancy across all walls
    """
    if not assigned_specs:
        logger.info(f"No wall objects assigned for stage {stage_suffix}")
        return [], 0.0

    logger.info(f"Processing {len(assigned_specs)} assigned wall objects wall by wall")
    walls_with_assignments = {getattr(spec, 'wall_id', None) for spec in assigned_specs}
    walls_with_assignments.discard(None)  # Remove None values

    # Create a lookup for wall data by wall_id
    wall_data_lookup = {wall_item.get("id", f"wall_{i}"): wall_item for i, wall_item in enumerate(all_wall_data)}

    processed_motifs = []
    solver_occupancy = 0.0
    for wall_id in walls_with_assignments:
        if wall_id not in wall_data_lookup:
            logger.warning(f"Wall {wall_id} not found in wall data, skipping")
            continue
        current_wall_data = wall_data_lookup[wall_id]
        specs_for_this_wall = [s for s in assigned_specs if getattr(s, 'wall_id', None) == wall_id]
        if not specs_for_this_wall:
            continue

        motifs_from_helper, wall_solver_occupancy = await _process_assigned_wall_objects(
            wall_session=wall_session,
            wall_id=str(wall_id),
            current_wall_data=current_wall_data,
            wall_specific_specs=specs_for_this_wall,
            cfg=cfg,
            output_dir=output_dir_override,
            vis_output_dir=vis_output_dir,
            room_description=room_description,
            model=model,
            scene_height=scene_height,
            stage_suffix=stage_suffix,
            room_bounds=room_bounds,
            solver_fallback=solver_fallback,
            scene=scene,
        )
        processed_motifs.extend(motifs_from_helper)

        # Accumulate solver occupancy across walls
        if wall_solver_occupancy > 0:
            solver_occupancy = max(solver_occupancy, wall_solver_occupancy)

    return processed_motifs, solver_occupancy


async def process_wall_objects(
    scene: Scene,
    cfg: DictConfig,
    output_dir_override: Path,
    room_description: str,
    model: ModelManager,
    visualizer: SceneVisualizer,
    current_room_plot: Any,
    sessions_dir: str,
    vis_output_dir: Path,
) -> Session:
    """
    Process wall objects for the scene in two stages: initial and extra objects.
    
    Args:
        scene: Scene object containing room and object data
        cfg: Configuration object
        output_dir_override: Output directory path
        room_description: Description of the room
        model: ModelManager instance for CLIP model
        visualizer: SceneVisualizer instance
        current_room_plot: Current scene plot
        sessions_dir: Directory for session storage
        vis_output_dir: Output directory path for visualization
        sessions_dir: Directory for session storage
        
    Returns:
        Wall session object
    """
    logger.info("Processing Wall Objects")

    if "wall" not in cfg.mode.object_types:
        logger.info("Wall objects not in processing types, skipping")
        wall_session = Session(PROMPT_DIR / "scene_prompts_wall.yaml", output_dir=str(sessions_dir))
        return wall_session
    
    # Calculate room bounds from room polygon for accurate wall object positioning
    room_bounds = None
    if hasattr(scene, 'room_polygon') and scene.room_polygon:
        try:
            bounds = scene.room_polygon.bounds  # Returns (minx, miny, maxx, maxy)
            room_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])  # (min_x, min_z, max_x, max_z)
            logger.debug(f"Calculated room bounds: {room_bounds}")
        except Exception as e:
            logger.warning(f"Could not extract room bounds from polygon: {e}")
    
    wall_session = Session(PROMPT_DIR / "scene_prompts_wall.yaml", output_dir=str(sessions_dir))
    processed_wall_motifs_all_stages: List[SceneMotif] = [] # Combined list for all wall objects
    threshold_percent = cfg.parameters.wall_object_generation.target_occupancy_percent
    MAX_WALL_ITERATION = cfg.parameters.wall_object_generation.max_iterations

    # Per-wall occupancy tracking - initialize with all walls at 0%
    wall_occupancy = {}  # wall_id -> current_occupancy_percent

    # Get initial wall data to set up tracking for all walls
    initial_wall_data = extract_wall_data(
        room_polygon=scene.room_polygon,
        scene_motifs=[],  # Empty to get base wall data
        door_location=scene.door_location,
        window_locations=getattr(scene, 'window_location', []),
    )

    # Initialize occupancy tracking for all walls
    for wall in initial_wall_data:
        wall_id = wall.get("id", f"wall_{initial_wall_data.index(wall)}")
        wall_occupancy[wall_id] = 0.0

    wall_iteration = 0

    # --- Wall Object Iteration Loop ---
    while wall_iteration < MAX_WALL_ITERATION:
        logger.info(f"Wall iteration {wall_iteration + 1}/{MAX_WALL_ITERATION}")

        # Check if all walls have reached target occupancy
        if _check_all_walls_complete(wall_occupancy, threshold_percent) and wall_iteration > 0:
            logger.info(f"All walls have reached target occupancy of {threshold_percent:.1f}%")
            break

        # Log current status for each wall
        status_lines = []
        for wall_id, occupancy in wall_occupancy.items():
            status_lines.append(f"Wall {wall_id}: {occupancy:.1f}%")
        if status_lines:
            logger.info(f"Current wall occupancy: {', '.join(status_lines)}, Target: {threshold_percent:.1f}%")
        else:
            logger.info(f"Current wall occupancy: 0.0%, Target: {threshold_percent:.1f}%")

        iteration_motifs = []
        solver_occupancy = 0.0

        # --- Stage 1: Process Initially Specified Wall Objects (only in first iteration) ---
        if wall_iteration == 0:
            logger.info("Stage 1: Processing initial wall objects")
            initial_wall_objects_spec = scene.scene_spec.wall_objects if scene.scene_spec and hasattr(scene.scene_spec, 'wall_objects') else []
            processed_initial_motifs: List[SceneMotif] = []

            if not initial_wall_objects_spec:
                logger.debug("No initial wall objects specified in scene spec")
            else:
                logger.debug(f"Found {len(initial_wall_objects_spec)} initial wall objects in spec")
                # Extract Wall Data (based on non-wall objects present so far)
                logger.debug("Extracting wall data for initial placement")
                current_non_wall_motifs = [m for m in scene.scene_motifs if m.object_type != ObjectType.WALL]

                initial_all_wall_data = extract_wall_data(
                    room_polygon=scene.room_polygon,
                    scene_motifs=current_non_wall_motifs, # Base availability on non-wall objects
                    door_location=scene.door_location,
                    window_locations=scene.window_location,
                )

                # Determine Eligible Walls for Initial Objects
                initial_eligible_walls = _get_eligible_walls(initial_all_wall_data, wall_occupancy, threshold_percent)

                if not initial_eligible_walls:
                    logger.warning("No walls eligible for initial wall objects (either no space or already at target occupancy)")
                else:
                    logger.info(f"Found {len(initial_eligible_walls)} eligible walls for initial placement: {list(initial_eligible_walls.keys())}")

                    # 1c. Assign Initial Objects to Walls
                    assigned_initial_specs = _assign_wall_objects_to_walls(
                        wall_session, initial_wall_objects_spec, initial_eligible_walls,
                        room_description, current_room_plot, "initial"
                    )

                    # Process Assigned Initial Objects using Helper
                    processed_initial_motifs, initial_solver_occupancy = await _process_wall_objects_by_wall(
                        assigned_initial_specs,
                        initial_all_wall_data,
                        wall_session,
                        cfg,
                        output_dir_override,
                        vis_output_dir,
                        room_description,
                        model,
                        scene.room_height,
                        "_initial",
                        room_bounds,
                        True,  # solver_fallback
                        scene
                    )
                    # Update global solver occupancy
                    solver_occupancy = max(solver_occupancy, initial_solver_occupancy)

                    # Add successfully processed initial motifs to the main scene BEFORE stage 2
                    current_room_plot = _add_motifs_and_visualize(
                        scene, processed_initial_motifs, visualizer, vis_output_dir,
                        wall_iteration, "initial"
                    )
                    if processed_initial_motifs:
                        processed_wall_motifs_all_stages.extend(processed_initial_motifs)
                        iteration_motifs.extend(processed_initial_motifs)

        # --- Stage 2: Generate and Process Extra Wall Objects ---
        processed_extra_motifs: List[SceneMotif] = []
        if "wall" in cfg.mode.extra_types and wall_iteration >= 0:
            logger.info("Stage 2: Generating and processing extra wall objects")
            # Re-evaluate Wall Data including now-placed initial wall objects
            logger.debug("Re-evaluating wall data with current wall objects placed")

            window_locs_2 = scene.window_location if hasattr(scene, "window_location") else []

            updated_all_wall_data = extract_wall_data(
                room_polygon=scene.room_polygon,
                scene_motifs=scene.scene_motifs,
                door_location=scene.door_location,
                window_locations=window_locs_2,
            )

            # Determine Eligible Walls for Extra Objects
            extra_eligible_walls = _get_eligible_walls(updated_all_wall_data, wall_occupancy, threshold_percent)

            if not extra_eligible_walls:
                logger.info("No walls have sufficient space remaining for extra wall objects or have reached target occupancy")
            else:
                logger.info(f"Found {len(extra_eligible_walls)} eligible walls for extra objects: {list(extra_eligible_walls.keys())}")

                # Generate Extra Wall Object Specs via VLM
                logger.info("Attempting to generate extra wall objects")
                generated_extra_specs: List[ObjectSpec] = []
                try:
                    wall_motifs = scene.get_motifs_by_types(ObjectType.WALL)
                    wall_motifs_data = [round_nested_values(m.to_gpt_dict(), 2) for m in wall_motifs] if wall_motifs else ["None"]

                    # Convert complex data to strings for VLM
                    wall_data_str = json.dumps(round_nested_values(updated_all_wall_data, 2))
                    wall_motifs_str = json.dumps(wall_motifs_data) if wall_motifs_data != ["None"] else "None"

                    generated_data_str = wall_session.send(
                        "wall_objects_extra",
                        {
                        "ROOM_TYPE": scene.room_type,
                        "WALL_DATA": wall_data_str,
                        "WALL_OBJECTS": wall_motifs_str
                    }, verbose=True, is_json=True
                    )
                    temp_spec = SceneSpec.from_json(generated_data_str)
                    if temp_spec.wall_objects:
                        generated_extra_specs = temp_spec.wall_objects
                        logger.info(f"Generated {len(generated_extra_specs)} extra wall object specifications")
                        if scene.scene_spec:
                            scene.scene_spec.add_objects(generated_extra_specs, "wall")
                except Exception as gen_e:
                    logger.error(f"Error during extra wall object generation VLM call: {gen_e}")
                    logger.error(traceback.format_exc())

                # Assign Generated Extra Objects to Walls
                if not generated_extra_specs:
                    logger.info("No extra wall objects generated")
                else:
                    assigned_extra_specs = _assign_wall_objects_to_walls(
                        wall_session, generated_extra_specs, extra_eligible_walls,
                        room_description, current_room_plot, "extra"
                    )

                    # Process Assigned Extra Objects using Helper
                    processed_extra_motifs, extra_solver_occupancy = await _process_wall_objects_by_wall(
                        assigned_extra_specs,
                        updated_all_wall_data,
                        wall_session,
                        cfg,
                        output_dir_override,
                        vis_output_dir,
                        room_description,
                        model,
                        scene.room_height,
                        "_extra",
                        room_bounds,
                        False,  # solver_fallback
                        scene
                    )
                    # Update global solver occupancy
                    solver_occupancy = max(solver_occupancy, extra_solver_occupancy)

            # Add successfully processed extra motifs to the main scene
            extra_plot = _add_motifs_and_visualize(
                scene, processed_extra_motifs, visualizer, vis_output_dir,
                wall_iteration, "extra"
            )
            if extra_plot:
                current_room_plot = extra_plot
            if processed_extra_motifs:
                processed_wall_motifs_all_stages.extend(processed_extra_motifs)
                iteration_motifs.extend(processed_extra_motifs)

        # Calculate per-wall occupancy for this iteration
        if iteration_motifs or solver_occupancy > 0:
            _update_wall_occupancy(scene, wall_occupancy, scene.door_location,
                                  scene.window_location if hasattr(scene, "window_location") else [])  # Occupied percentage

            # Print updated occupancy for each wall
            occupancy_updates = []
            for wall_id, occupancy in wall_occupancy.items():
                occupancy_updates.append(f"Wall {wall_id}: {occupancy:.1f}%")
            if occupancy_updates:
                logger.info(f"Updated wall occupancy: {', '.join(occupancy_updates)}")
        else:
            logger.info(f"No wall motifs processed in iteration {wall_iteration + 1}")

        wall_iteration += 1

        if output_dir_override:
            save_scene(scene, output_dir_override)

    logger.info(f"Finished wall processing after {wall_iteration} iterations")

    final_status = []
    for wall_id, occupancy in wall_occupancy.items():
        final_status.append(f"Wall {wall_id}: {occupancy:.1f}%")

    if final_status:
        all_complete = _check_all_walls_complete(wall_occupancy, threshold_percent)
        logger.info(f"Final wall occupancy: {', '.join(final_status)}, Target: {threshold_percent:.1f}%")
        if all_complete:
            logger.info("All walls reached target occupancy!")
        # else:
        #     logger.info("Some walls did not reach target occupancy")

    logger.info(f"{len(processed_wall_motifs_all_stages)} wall motifs added: {[m.id for m in processed_wall_motifs_all_stages]}")
    return wall_session