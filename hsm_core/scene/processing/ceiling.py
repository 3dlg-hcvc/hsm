"""
Ceiling Object Processing Module
"""

import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from shapely.geometry import Polygon
from omegaconf import DictConfig

from hsm_core.scene.ablation import create_individual_scene_motifs_with_analysis
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.core.motif import SceneMotif
from hsm_core.scene.core.manager import Scene
from hsm_core.config import PROMPT_DIR
from hsm_core.scene.core.spec import SceneSpec
from hsm_core.scene.visualization.visualization import SceneVisualizer
from hsm_core.scene.processing.generate_scene_motif import process_scene_motifs
from hsm_core.scene.validation.validate import validate_arrangement_smc, validate_furniture_layout
from hsm_core.utils import get_logger
from hsm_core.scene.processing.processing_helpers import (
    prepare_ceiling_solver_inputs,
    run_solver_and_update_motifs,
    update_ceiling_motif_from_solver,
    run_spatial_optimization_for_stage,
    filter_motifs_needing_optimization,
    CEILING_Y_OFFSET
)

from hsm_core.vlm.gpt import Session, extract_json

logger = get_logger('scene.processing.ceiling')
from hsm_core.retrieval.model.model_manager import ModelManager
from hsm_core.scene.io.export import save_scene


def position_ceiling_motifs_with_vlm(
    ceiling_session: Session,
    motifs: List[SceneMotif],
    scene: Scene,
    room_polygon: Polygon,
    updated_plot: Any,
    motif_type: str
) -> None:
    """
    Position ceiling motifs using VLM positioning.

    Args:
        ceiling_session: Session for LLM communication
        motifs: List of motifs to position
        scene: Scene object containing room geometry
        room_polygon: Room polygon for validation
        updated_plot: Current scene plot for visualization
        motif_type: Type of motifs ("initial" or "extra") for logging
    """
    logger.info(f"Getting detailed positioning for {motif_type} ceiling objects")
    try:
        position_params = {
            "AREAS": [f"{m.id}:{m.description}:{m.extents}" for m in motifs],
            "ROOM_VERTICES": str(scene.room_vertices)
        }
        placement_images = [updated_plot] if updated_plot else []
        detailed_position_data = ceiling_session.send_with_validation(
            "populate_room_provided",
            position_params,
            lambda response: validate_furniture_layout(response, room_polygon, motifs),
            verbose=True,
            images=placement_images,
            is_json=True
        )

        positioned_count = 0
        for item in json.loads(extract_json(detailed_position_data)).get("positions", []):
            motif_id = item.get("id")
            llm_pos_2d = item.get("position")
            rotation = item.get("rotation", 0)
            matching_motif = next((m for m in motifs if m.id == motif_id), None)

            if matching_motif and llm_pos_2d and len(llm_pos_2d) >= 2:
                pos = (llm_pos_2d[0], scene.room_height + CEILING_Y_OFFSET, llm_pos_2d[1])
                matching_motif.position = pos
                matching_motif.rotation = rotation
                matching_motif.object_type = ObjectType.CEILING
                positioned_count += 1
                logger.debug(f"  - Positioned {motif_type} ceiling motif {motif_id} at {pos} with rotation {rotation}")

        logger.info(f"VLM positioned {positioned_count} {motif_type} ceiling motifs")
    except Exception as e:
        logger.warning(f"Failed to get VLM positioning for {motif_type} ceiling objects: {e}")
        # Fallback to default positioning
        for motif in motifs:
            motif.position = (motif.position[0], scene.room_height + CEILING_Y_OFFSET, motif.position[2])
            motif.object_type = ObjectType.CEILING


def run_ceiling_solver_and_update_scene(
    scene: Scene,
    cfg: DictConfig,
    motifs_to_solve: List[SceneMotif],
    processed_motifs_this_iteration: List[SceneMotif],
    room_polygon: Polygon,
    output_dir_override: Path,
    vis_output_dir: Path,
    iteration_occupancy: float,
    motif_type: str,
    solver_fallback: bool,
    update_global_occupancy: bool = False
) -> Tuple[List[SceneMotif], float]:
    """
    Run solver for ceiling motifs and update scene.

    Args:
        scene: Scene object
        cfg: Configuration
        motifs_to_solve: Motifs to run solver on
        processed_motifs_this_iteration: List to extend with solved motifs
        room_polygon: Room geometry
        output_dir_override: Output directory
        iteration_occupancy: Current iteration occupancy
        motif_type: Type of motifs ("initial" or "extra") for logging
        solver_fallback: Whether to use solver fallback
        update_global_occupancy: Whether to update global occupancy with max()

    Returns:
        Tuple of (solved_motifs, updated_occupancy)
    """
    if not motifs_to_solve or not cfg.mode.use_solver:
        # If solver is disabled, add all motifs to scene
        scene.add_motifs(motifs_to_solve)
        processed_motifs_this_iteration.extend(motifs_to_solve)
        logger.info(f"Added {len(motifs_to_solve)} {motif_type} ceiling motifs to scene (solver disabled)")
        return motifs_to_solve, iteration_occupancy

    # Get existing ceiling motifs as fixed obstacles (except for initial which has none)
    fixed_motifs = []
    if motif_type != "initial":
        fixed_motifs = [
            m for m in scene.scene_motifs
            if m.object_type == ObjectType.CEILING and m not in motifs_to_solve
        ]

    solver_inputs = prepare_ceiling_solver_inputs(
        motifs_to_place=motifs_to_solve,
        fixed_motifs=fixed_motifs
    )

    update_context = {"room_height": scene.room_height}
    placed_motifs, occupancy = run_solver_and_update_motifs(
        solver_inputs=solver_inputs,
        geometry=room_polygon,
        target_motifs_list=motifs_to_solve,
        output_dir=str(vis_output_dir),
        subfix=f"ceiling_{motif_type}",
        enable_solver=cfg.mode.use_solver,
        update_func=update_ceiling_motif_from_solver,
        update_context=update_context,
        solver_fallback=solver_fallback
    )

    if update_global_occupancy:
        iteration_occupancy = max(iteration_occupancy, occupancy)

    solved_ids = {m["id"] for m in placed_motifs} if placed_motifs else set()
    solved_motifs = [m for m in motifs_to_solve if m.id in solved_ids]
    processed_motifs_this_iteration.extend(solved_motifs)

    unsolved_motifs = [m for m in motifs_to_solve if m.id not in solved_ids]
    if unsolved_motifs:
        logger.info(
            f"Discarding {len(unsolved_motifs)} {motif_type} ceiling motif(s) not placed by solver: "
            f"{[m.id for m in unsolved_motifs]}"
        )

    if solved_motifs:
        scene.add_motifs(solved_motifs)
        logger.info(f"Added {len(solved_motifs)} successfully placed {motif_type} ceiling motifs to scene")

    logger.info(f"{motif_type.capitalize()} ceiling solver completed. Occupancy: {occupancy:.3f}")

    # Run spatial optimization
    motifs_needing_optimization = filter_motifs_needing_optimization(solved_motifs)
    if motifs_needing_optimization:
        run_spatial_optimization_for_stage(
            scene=scene,
            cfg=cfg,
            current_stage_motifs=motifs_needing_optimization,
            object_type=ObjectType.CEILING,
            output_dir=output_dir_override,
            stage_name=motif_type
        )

    return solved_motifs, iteration_occupancy if update_global_occupancy else occupancy


async def process_ceiling_objects(
    scene: Scene,
    cfg: DictConfig,
    output_dir_override: Path,
    room_description: str,
    model: ModelManager,
    visualizer: SceneVisualizer,
    room_polygon: Polygon,
    current_room_plot: Any,
    sessions_dir: str,
    vis_output_dir: Path,
) -> Session:
    """
    Process ceiling objects for the scene.
    
    Args:
        scene: Scene object containing room and object data
        cfg: Configuration object
        output_dir_override: Output directory path
        room_description: Description of the room
        model: ModelManager instance for CLIP model
        visualizer: SceneVisualizer instance
        room_polygon: Shapely polygon representing room geometry
        current_room_plot: Current scene plot
        sessions_dir: Directory for session storage
        
    Returns:
        Session object used for ceiling processing
    """
    logger.info("Processing Ceiling Objects")

    if "ceiling" not in cfg.mode.object_types:
        logger.info("Ceiling objects not in processing types, skipping")
        dummy_session = Session(str(PROMPT_DIR / "scene_prompts_ceiling.yaml"))
        dummy_session.output_dir = sessions_dir
        return dummy_session
    
    ceiling_session = Session(str(PROMPT_DIR / "scene_prompts_ceiling.yaml"))
    ceiling_session.output_dir = sessions_dir

    MAX_CEILING_ITERATION = cfg.parameters.ceiling_object_generation.max_iterations
    target_occupancy_percent = cfg.parameters.ceiling_object_generation.target_occupancy_percent
    cumulative_ceiling_occupancy = 0.0
    ceiling_iteration = 0

    while ceiling_iteration < MAX_CEILING_ITERATION:
        if ceiling_iteration > 0:
            logger.info(f"--- Ceiling Object Iteration {ceiling_iteration} ---")
            logger.info(f"Current ceiling occupancy: {cumulative_ceiling_occupancy:.1f}%, Target: {target_occupancy_percent:.1f}%")
        else:
            logger.info(f"--- Ceiling Object Iteration {ceiling_iteration} ---")

        if cumulative_ceiling_occupancy >= target_occupancy_percent and ceiling_iteration > 0:
            logger.info(f"Target ceiling occupancy of {target_occupancy_percent:.1f}% has been reached")
            break

        iteration_occupancy = 0.0
        processed_motifs_this_iteration: List[SceneMotif] = []

        initial_ceiling_motifs = []
        ceiling_analysis = None
        motif_spec = {}

        if ceiling_iteration == 0 and scene.scene_spec.ceiling_objects:
            logger.info("Processing initial ceiling objects")

            ceiling_analysis = ceiling_session.send_with_validation(
            "populate_surface_motifs",
            {
                "room_type": scene.room_description,
                "ceiling_objects": [obj.to_dict() for obj in scene.scene_spec.ceiling_objects],
                "room_vertices": str(scene.room_vertices)
            },
            lambda response: validate_arrangement_smc(
                response,
                [obj.id for obj in scene.scene_spec.ceiling_objects],
                scene.scene_spec.ceiling_objects
            ),
            is_json=True,
            verbose=True,
                    )

            if ceiling_analysis:
                motif_spec = json.loads(extract_json(ceiling_analysis))

            if not cfg.mode.use_scene_motifs:
                analysis_str = create_individual_scene_motifs_with_analysis([obj.to_dict() for obj in scene.scene_spec.ceiling_objects], motif_spec)
                motif_spec = json.loads(analysis_str)

            initial_ceiling_motifs, _ = await process_scene_motifs(
                scene.scene_spec.ceiling_objects,
                motif_spec,
                output_dir_override,
                room_description,
                model,
                object_type=ObjectType.CEILING,
            )

            if initial_ceiling_motifs:
                position_ceiling_motifs_with_vlm(
                    ceiling_session, initial_ceiling_motifs, scene, room_polygon, current_room_plot, "initial"
                )

                initial_ceiling_motifs_solved, iteration_occupancy = run_ceiling_solver_and_update_scene(
                    scene=scene,
                    cfg=cfg,
                    motifs_to_solve=initial_ceiling_motifs,
                    processed_motifs_this_iteration=processed_motifs_this_iteration,
                    room_polygon=room_polygon,
                    output_dir_override=output_dir_override,
                    vis_output_dir=vis_output_dir,
                    iteration_occupancy=iteration_occupancy,
                    motif_type="initial",
                    solver_fallback=True
                )

        extra_ceiling_motifs = []
        if "ceiling" in cfg.mode.extra_types and ceiling_iteration >= 0:
            logger.info("Processing extra ceiling objects")

            objects_response = ceiling_session.send(
                "ceiling_objects_extra",
                {
                    "room_type": scene.room_type,
                    "ceiling_objects": str([motif for motif in scene.scene_motifs if motif.object_type == ObjectType.CEILING])
                },
                is_json=True,
                verbose=True
            )
            extra_ceiling_objects_raw = SceneSpec.from_json(objects_response)
            extra_ceiling_spec = scene.scene_spec.add_objects(extra_ceiling_objects_raw.ceiling_objects, "ceiling")

            if extra_ceiling_spec.ceiling_objects:
                motif_spec = {}
                ceiling_analysis = ceiling_session.send_with_validation(
                    "populate_surface_motifs",
                    {
                        "room_type": scene.room_type,
                        "ceiling_objects": [obj.to_dict() for obj in extra_ceiling_spec.ceiling_objects],
                        "room_vertices": str(scene.room_vertices)
                    },
                    lambda response: validate_arrangement_smc(
                        response,
                        [obj.id for obj in extra_ceiling_spec.ceiling_objects],
                        extra_ceiling_spec.ceiling_objects,
                        scene.scene_spec.ceiling_objects,
                        scene.get_motifs_by_types(ObjectType.CEILING) or []
                    ),
                    is_json=True,
                    verbose=True,
                )
                motif_spec = json.loads(extract_json(ceiling_analysis))

                if not cfg.mode.use_scene_motifs:
                    analysis_str = create_individual_scene_motifs_with_analysis([obj.to_dict() for obj in extra_ceiling_spec.ceiling_objects], motif_spec)
                    motif_spec = json.loads(analysis_str)

                extra_ceiling_motifs, _ = await process_scene_motifs(
                    extra_ceiling_spec.ceiling_objects,
                    motif_spec,
                    output_dir_override,
                    room_description,
                    model,
                    object_type=ObjectType.CEILING,
                )

                if extra_ceiling_motifs:
                    position_ceiling_motifs_with_vlm(
                        ceiling_session, extra_ceiling_motifs, scene, room_polygon, current_room_plot, "extra"
                    )

                    extra_ceiling_motifs_solved, iteration_occupancy = run_ceiling_solver_and_update_scene(
                        scene=scene,
                        cfg=cfg,
                        motifs_to_solve=extra_ceiling_motifs,
                        processed_motifs_this_iteration=processed_motifs_this_iteration,
                        room_polygon=room_polygon,
                        output_dir_override=output_dir_override,
                        vis_output_dir=vis_output_dir,
                        iteration_occupancy=iteration_occupancy,
                        motif_type="extra",
                        solver_fallback=False,
                        update_global_occupancy=True
                    )

        if processed_motifs_this_iteration:
            cumulative_ceiling_occupancy = max(cumulative_ceiling_occupancy, iteration_occupancy)
            logger.info(f"Iteration {ceiling_iteration} completed. Occupancy: {iteration_occupancy:.3f}, Cumulative: {cumulative_ceiling_occupancy:.3f}%")
        else:
            logger.info(f"Iteration {ceiling_iteration} completed. No motifs processed")

        ceiling_iteration += 1

        if output_dir_override:
            save_scene(scene, output_dir_override)

    logger.info(f"Finished ceiling object processing after {ceiling_iteration} iterations. Cumulative occupancy: {cumulative_ceiling_occupancy:.1f}%")
    return ceiling_session 