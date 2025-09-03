"""
Large Object Processing Module
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from shapely.geometry import Polygon

from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.motif import SceneMotif
from hsm_core.scene.manager import Scene
from hsm_core.scene.spec import ObjectSpec, SceneSpec
from hsm_core.scene.visualization import SceneVisualizer
from hsm_core.scene.generate_scene_motif import process_scene_motifs
from hsm_core.scene.validate import validate_arrangement_smc, validate_furniture_layout
from hsm_core.scene.ablation import create_individual_scene_motifs_with_analysis
from hsm_core.config import PROMPT_DIR

from hsm_core.vlm.gpt import Session, extract_json
from hsm_core.retrieval.model.model_manager import ModelManager
from hsm_core.scene.processing_helpers import (
    prepare_large_solver_inputs, 
    run_solver_and_update_motifs, 
    update_large_motif_from_solver,
    run_spatial_optimization_for_stage,
    filter_motifs_needing_optimization,
)

def _create_large_session(sessions_dir: str) -> Session:
    """Create a session for large object processing."""
    session = Session(str(PROMPT_DIR / "scene_prompts_large.yaml"))
    session.output_dir = sessions_dir
    return session


async def _process_motif_stage(
    large_session: Session,
    scene_objects: List[ObjectSpec],
    room_description: str,
    output_dir_override: Path,
    model: ModelManager,
    cfg: DictConfig,
    scene: Scene,
    combined_fig: Any,
    updated_plot: Any,
    room_polygon: Polygon,
    is_extra: bool = False,
    existing_motifs: Optional[List[SceneMotif]] = None,
) -> Tuple[List[SceneMotif], Any]:
    """Process a stage of motifs (initial or extra)."""
    motif_spec = {}

    # Determine validation function parameters based on whether this is extra iteration
    if is_extra:
        existing_large_motifs = scene.get_motifs_by_types(ObjectType.LARGE) or []
        validation_func = lambda response: validate_arrangement_smc(
            response,
            [obj.id for obj in scene_objects],
            scene_objects,
            scene.scene_spec.large_objects,
            existing_large_motifs,
        )
        prompt_data = {
            "room_type": scene.room_type,
            "large_furniture": [obj.to_dict() for obj in scene_objects],
            "existing_motifs": [motif.to_gpt_dict() for motif in existing_motifs] if existing_motifs else [],
        }
    else:
        validation_func = lambda response: validate_arrangement_smc(
            response,
            [obj.id for obj in scene_objects],
            scene.scene_spec.large_objects,
        )
        prompt_data = {
            "room_details": scene.room_details,
            "room_type": room_description,
            "large_furniture": [obj.to_dict() for obj in scene_objects],
        }

    analysis = large_session.send_with_validation(
        "populate_surface_motifs",
        prompt_data,
        validation_func,
        json=True,
        verbose=True,
    )
    motif_spec = json.loads(extract_json(analysis))

    if not cfg.mode.use_scene_motifs:
        analysis_str = create_individual_scene_motifs_with_analysis(
            [obj.to_dict() for obj in scene_objects], motif_spec
        )
        motif_spec = json.loads(analysis_str)

    if not motif_spec:
        print("Warning: No motif spec found")
        return [], large_session

    scene_motifs, combined_fig_result = await process_scene_motifs(
        scene_objects,
        motif_spec,
        output_dir_override,
        room_description,
        model,
        object_type=ObjectType.LARGE,
    )

    if not scene_motifs:
        print("No scene motifs to process")
        return [], large_session


    images = [combined_fig_result, updated_plot] if is_extra else [combined_fig_result, combined_fig]
    analysis = large_session.send_with_validation(
        "populate_room_provided",
        {"MOTIFS": [str(motif.to_gpt_dict()) for motif in scene_motifs]},
        lambda response: validate_furniture_layout(response, room_polygon, scene_motifs),
        json=True,
        verbose=True,
        images=images,
    )


    if analysis:
        motif_spec = json.loads(extract_json(analysis))
        _apply_vlm_positions_to_motifs(motif_spec, scene_motifs)

    return scene_motifs, large_session


def _apply_vlm_positions_to_motifs(motif_spec: Dict[str, Any], scene_motifs: List[SceneMotif]) -> None:
    """Apply VLM positions to scene motifs."""
    for item in motif_spec["positions"]:
        matching_motif = next((motif for motif in scene_motifs if motif.id == item["id"]), None)
        if matching_motif:
            pos = (item["position"][0], 0, item["position"][1])
            matching_motif.position = pos
            matching_motif.rotation = item["rotation"]
            matching_motif.object_type = ObjectType.LARGE
            matching_motif.ignore_collision = item.get("ignore_collision", False)
            matching_motif.wall_alignment = item.get("wall_alignment", False)
            matching_motif.wall_alignment_id = item.get("wall_alignment_id", None)
            if matching_motif.wall_alignment:
                print(f"  - Stored wall alignment for {matching_motif.id}: wall_id={matching_motif.wall_alignment_id}")


async def process_large_objects(
    scene: Scene,
    cfg: DictConfig,
    output_dir_override: Path,
    room_description: str,
    model: ModelManager,
    visualizer: SceneVisualizer,
    room_polygon: Polygon,
    initial_plot: Any,
    project_root: Path,
    sessions_dir: str
) -> Tuple[Any, Session]:
    """
    Process large objects for the scene.
    
    Args:
        scene: Scene object containing room and object data
        cfg: Configuration object
        output_dir_override: Output directory path
        room_description: Description of the room
        model: ModelManager instance for CLIP model
        visualizer: SceneVisualizer instance
        room_polygon: Shapely polygon representing room geometry
        initial_plot: Initial room plot
        project_root: Path to project root
        sessions_dir: Directory for session storage
        
    Returns:
        Tuple of (updated_plot, session)
    """
    print("Processing Large Objects...")
    
    MAX_ITERATION = cfg.parameters.large_object_generation.max_iterations
    occupancy_percent = 100.0
    iteration = 0
    updated_plot = None
    solver_fallback = True
    cumulative_occupancy = 0.0
    
    if "large" not in cfg.mode.object_types:
        print("Large objects not in processing types, skipping...")
        # Create a dummy session for consistency
        dummy_session = _create_large_session(sessions_dir)
        return updated_plot, dummy_session
    
    if not hasattr(scene, 'scene_spec') or not scene.scene_spec:
        decompose_session = _create_large_session(sessions_dir)
        
        objects_response = decompose_session.send(
            "requirements_decompose", {"room_description": room_description}, json=True, verbose=True
        )
        scene.scene_spec = SceneSpec.from_json(objects_response, required=True)
    
    while occupancy_percent > cfg.parameters.large_object_generation.target_occupancy_percent and iteration < MAX_ITERATION:
        analysis = None  # Reset analysis for each iteration
        current_stage_motifs = []  # Track motifs for this stage
        current_stage_motifs_solved: List[SceneMotif] = []  # Ensure variable is always defined
        
        if iteration == 0:
            large_session = _create_large_session(sessions_dir)
            current_stage_motifs, large_session = await _process_motif_stage(
                large_session,
                scene.scene_spec.large_objects,
                room_description,
                output_dir_override,
                model,
                cfg,
                scene,
                initial_plot,
                updated_plot,
                room_polygon,
            )
            
        elif "large" not in cfg.mode.extra_types:
            break
        else:
            print("=" * 50)
            print(f"Starting large object iteration {iteration}")
            print(
                f"Adding more large objects to the scene, occupancy percent: {occupancy_percent}, cumulative occupancy: {cumulative_occupancy:.3f}"
            )
            large_session = _create_large_session(sessions_dir)

            objects_response = large_session.send(
                "large_furniture_extra",
                {"room_type": scene.room_type, "large_furniture": str(scene.scene_motifs)},
                json=True,
                verbose=True,
                images=[updated_plot],
            )
            extra_scene_spec_raw = SceneSpec.from_json(objects_response)
            extra_scene_spec = scene.scene_spec.add_objects(extra_scene_spec_raw.large_objects, "large")

            current_stage_motifs, large_session = await _process_motif_stage(
                large_session,
                extra_scene_spec.large_objects,
                room_description,
                output_dir_override,
                model,
                cfg,
                scene,
                initial_plot,
                updated_plot,
                room_polygon,
                is_extra=True,
                existing_motifs=scene.get_motifs_by_types(ObjectType.LARGE) or [],
            )

            solver_fallback = False

        if current_stage_motifs and cfg.mode.use_solver:
            print(f"Running solver for {len(current_stage_motifs)} large motifs (iteration {iteration})")
            
            existing_large_motifs = [
                m for m in scene.scene_motifs 
                if m.object_type == ObjectType.LARGE and m not in current_stage_motifs
            ]
            
            solver_inputs = prepare_large_solver_inputs(
                motifs_to_place=current_stage_motifs,
                fixed_motifs=existing_large_motifs,
                scene=scene
            )

            surface_geometry = Polygon(scene.room_vertices)

            placed_motifs, iteration_occupancy = run_solver_and_update_motifs(
                solver_inputs=solver_inputs,
                geometry=surface_geometry,
                target_motifs_list=current_stage_motifs,
                output_dir=str(output_dir_override),
                subfix=f"large_iteration_{iteration}",
                enable_solver=cfg.mode.use_solver,
                update_func=update_large_motif_from_solver,
                update_context=None,
                solver_fallback=solver_fallback
            )

            if placed_motifs:  # Only update if solver actually placed objects
                cumulative_occupancy = max(cumulative_occupancy, iteration_occupancy)
                occupancy_percent = (1.0 - cumulative_occupancy) * 100.0
                print(f"Updated occupancy: {iteration_occupancy:.3f}, cumulative: {cumulative_occupancy:.3f}, occupancy: {occupancy_percent:.1f}%")
            else:
                print(f"No objects placed by solver, keeping previous occupancy: {occupancy_percent:.1f}%")

            solved_ids = {m["id"] for m in placed_motifs} if placed_motifs else set()
            current_stage_motifs_solved = [m for m in current_stage_motifs if m.id in solved_ids]

            unsolved_motifs = [m for m in current_stage_motifs if m.id not in solved_ids]
            if unsolved_motifs:
                print(
                    f"Discarding {len(unsolved_motifs)} motif(s) that were not placed by the solver: "
                    f"{[m.id for m in unsolved_motifs]}"
                )

            if current_stage_motifs_solved:
                scene.add_motifs(current_stage_motifs_solved)
                print(f"Added {len(current_stage_motifs_solved)} successfully placed motifs to scene")

            print(f"Solver completed for iteration {iteration}. Occupancy: {iteration_occupancy:.3f}")
        else:
            current_stage_motifs_solved = current_stage_motifs
            scene.add_motifs(current_stage_motifs_solved)
            print(f"Added {len(current_stage_motifs_solved)} motifs to scene (solver disabled)")

        motifs_needing_optimization = filter_motifs_needing_optimization(current_stage_motifs_solved)
        if motifs_needing_optimization:
            stage_name = f"iteration_{iteration}"
            run_spatial_optimization_for_stage(
                scene=scene,
                cfg=cfg,
                current_stage_motifs=motifs_needing_optimization,
                object_type=ObjectType.LARGE,
                output_dir=output_dir_override,
                stage_name=stage_name
            )

        updated_plot, _ = visualizer.visualize(
            output_path=str(output_dir_override / f"large_iteration_{iteration}_plot.png"),
            add_grid_markers=True
        )

        scene.export(output_dir_override / f"room_scene.glb", recreate_scene=True)
        scene.save(output_dir_override)
        iteration += 1

    print(f"Finished large object processing. Iterations: {iteration}")
    return updated_plot, large_session