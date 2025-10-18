"""
Small Object Placer Module

This module handles the placement and population of small objects in the scene.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import traceback
from typing import Optional
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from hsm_core.vlm.gpt import Session, extract_json
from hsm_core.config import PROMPT_DIR
from hsm_core.scene.validation.validate import validate_small_object_response, validate_arrangement_smc
from hsm_core.scene.processing.small_object_helpers import (
    collect_surface_data,
    optimize_small_objects,
    populate_furniture,
    update_small_motifs_from_constrained_layout,
    clean_layer_info
)
from hsm_core.scene.core.motif import SceneMotif, filter_motifs_by_types
from hsm_core.scene.core.spec import ObjectSpec
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.processing.generate_scene_motif import process_scene_motifs
from hsm_core.utils import get_logger

logger = get_logger('scene.small_object_processing')

def setup_small_objects_population(scene) -> bool:
    """Setup and validate conditions for small object population."""
    if scene.scene is None:
        scene.create_scene()

    if not scene.scene_spec:
        logger.warning("SceneSpec is not initialized. Cannot populate small objects")
        return False
    return True


def get_constrained_small_objects(scene) -> tuple[list, set]:
    """Get constrained small objects and their parent motifs."""
    constrained_small_objects = [
        obj for obj in scene.scene_spec.small_objects
        if obj.parent_object is not None and obj.required
    ]
    constrained_parent_motifs = {obj.parent_object for obj in constrained_small_objects}

    logger.info("=" * 60)
    logger.info("Processing Constrained Small Objects")
    logger.debug(f"Found {len(constrained_small_objects)} constrained small objects: {constrained_small_objects}")

    return constrained_small_objects, constrained_parent_motifs


def build_parent_name_to_id_map(motif: SceneMotif, constrained_parent_motifs: set = None) -> dict[str, int]:
    """Build mapping from instance names to object IDs for a motif."""
    parent_name_to_id_map = {}
    name_counts = {}

    for spec_in_motif in motif.object_specs:
        # Skip if we have constraints and this object is not a constrained parent
        if constrained_parent_motifs is not None and spec_in_motif.id not in constrained_parent_motifs:
            continue

        base_name = spec_in_motif.name
        if base_name in name_counts:
            name_counts[base_name] += 1
            instance_name = f"{base_name}_{name_counts[base_name]}"
        else:
            name_counts[base_name] = 1
            instance_name = base_name  # First instance keeps the original name
        parent_name_to_id_map[instance_name] = spec_in_motif.id

    return parent_name_to_id_map


def group_instances_by_base_name(parent_name_to_id_map: dict[str, int], scene) -> dict[str, list[str]]:
    """Group instance names by their base name using batch object spec lookups."""
    base_name_to_instance_names = {}

    for instance_name, obj_id in parent_name_to_id_map.items():
        parent_obj_spec = scene.scene_spec.get_object_by_id(obj_id)
        if not parent_obj_spec:
            continue
        base_name = parent_obj_spec.name
        if base_name not in base_name_to_instance_names:
            base_name_to_instance_names[base_name] = []
        base_name_to_instance_names[base_name].append(instance_name)

    return base_name_to_instance_names


def build_object_specs_lookup(scene) -> dict[int, tuple[SceneMotif, ObjectSpec]]:
    """Build lookup dictionary for object specifications."""
    all_object_specs: dict[int, tuple[SceneMotif, ObjectSpec]] = {}
    for scene_motif in scene.scene_motifs:
        for obj_spec in scene_motif.object_specs:
            if obj_spec.id:  # Check if ID exists and is valid
                try:
                    all_object_specs[int(obj_spec.id)] = (scene_motif, obj_spec)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping object spec with non-integer ID {obj_spec.id} in motif {scene_motif.id}")

    logger.debug(f"Built lookup with {len(all_object_specs)} entries")
    return all_object_specs


async def process_constrained_small_objects(
    scene,
    constrained_small_objects: list,
    constrained_parent_motifs: set,
    all_object_specs: dict[int, tuple[SceneMotif, ObjectSpec]],
    output_dir: str,
    cfg,
    model,
    vis_output_dir
) -> set[str]:
    """Process constrained small objects for motifs that have specific assignments."""
    processed_motifs: set[str] = set()

    # Find motifs that have constrained small objects
    constrained_motifs = []
    for motif in scene.scene_motifs:
        parent_object_ids = {spec.id for spec in motif.object_specs}
        if parent_object_ids.intersection(constrained_parent_motifs):
            constrained_motifs.append(motif)

    logger.info(f"Found {len(constrained_motifs)} motifs with constrained small objects: {[m.id for m in constrained_motifs]}")

    for motif in constrained_motifs:
        # Get all small objects for parents in this motif
        parent_object_ids = {spec.id for spec in motif.object_specs}
        relevant_small_specs: list[ObjectSpec] = [
            obj for obj in constrained_small_objects
            if obj.parent_object in parent_object_ids
        ]

        if not relevant_small_specs:
            logger.debug(f"Motif {motif.id} has no relevant small objects")
            continue

        # Construct parent_name_to_id_map for the current motif
        constrained_motif_parent_name_to_id_map = build_parent_name_to_id_map(motif, constrained_parent_motifs)

        # Collect and process surface data
        layer_data = await collect_constrained_surface_data(
            scene, motif, constrained_motif_parent_name_to_id_map, constrained_parent_motifs, output_dir
        )

        if not layer_data:
            logger.warning(f"No layer data collected for scene motif {motif.id}. Skipping constrained small object processing")
            continue

        # Process small objects for this motif
        try:
            await process_single_motif_constrained_small_objects(
                scene, motif, relevant_small_specs, constrained_motif_parent_name_to_id_map,
                layer_data, output_dir, cfg, model, all_object_specs, vis_output_dir
            )
            processed_motifs.add(motif.id)
        except Exception as e:
            logger.info(f"Skipping scene motif {motif.id} small objects constrained population due to error: {e}")
            logger.debug(traceback.format_exc())
            continue

    return processed_motifs


async def collect_constrained_surface_data(
    scene, motif: SceneMotif, parent_name_to_id_map: dict[str, int],
    constrained_parent_motifs: set, output_dir: str
) -> Optional[dict]:
    """Collect surface data for constrained small objects."""
    # Get instance names for constrained parents only
    constrained_parent_instance_names = [
        name for name, obj_id in parent_name_to_id_map.items()
        if obj_id in constrained_parent_motifs
    ]

    if not constrained_parent_instance_names:
        return None

    # Collect surface data for constrained parents
    layer_data, _, layer_fig = collect_surface_data(
        large_object_names=constrained_parent_instance_names,
        motif=motif,
        output_dir=output_dir,
        try_ransac=False
    )

    if layer_fig:
        plt.close(layer_fig)

    # Remap layer_data keys to use instance names
    if layer_data:
        remapped_layer_data = {}
        base_name_to_instance_names = group_instances_by_base_name(parent_name_to_id_map, scene)

        # Create case-insensitive mapping for layer data keys
        layer_data_case_map = {key.lower(): key for key in layer_data.keys()}
        base_name_case_map = {name.lower(): name for name in base_name_to_instance_names.keys()}

        # Remap layer data using instance names with case-insensitive matching
        for layer_key, layer_info in layer_data.items():
            # Try exact match first
            if layer_key in base_name_to_instance_names:
                instance_names = base_name_to_instance_names[layer_key]
                for instance_name in instance_names:
                    if parent_name_to_id_map.get(instance_name) in constrained_parent_motifs:
                        remapped_layer_data[instance_name] = layer_info
            # Fall back to case-insensitive match
            elif layer_key.lower() in base_name_case_map:
                actual_base_name = base_name_case_map[layer_key.lower()]
                instance_names = base_name_to_instance_names[actual_base_name]
                for instance_name in instance_names:
                    if parent_name_to_id_map.get(instance_name) in constrained_parent_motifs:
                        remapped_layer_data[instance_name] = layer_info

        return remapped_layer_data

    return None


async def process_single_motif_constrained_small_objects(
    scene, motif: SceneMotif, relevant_small_specs: list,
    constrained_motif_parent_name_to_id_map: dict[str, int],
    layer_data: dict, output_dir: str, cfg, model, all_object_specs: dict, vis_output_dir: Path
) -> None:
    """Process constrained small objects for a single motif."""
    small_obj_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))

    # Prepare layer_data for VLM using clean_layer_info
    layer_data_for_llm = clean_layer_info(layer_data)

    # Get layered structure for these constrained small objects
    parent_names_str = [
        name for name, obj_id in constrained_motif_parent_name_to_id_map.items()
        if obj_id in {obj.parent_object for obj in relevant_small_specs}
    ]

    objects_response_str = small_obj_session.send_with_validation(
        "small_objects_layered",
        {
            "small_objects": [obj.to_gpt_dict() for obj in relevant_small_specs],
            "motif_description": str(motif.description),
            "large_furniture": str(parent_names_str),
            "room_type": scene.room_description,
            "layer_info": json.dumps(layer_data_for_llm)
        },
        lambda resp: validate_small_object_response(resp,
                                                  relevant_small_specs,
                                                  parent_names_str,
                                                  layer_data),
        is_json=True,
        verbose=True
    )

    parsed_llm_data = json.loads(extract_json(objects_response_str))
    added_small_objects_spec_container = scene.scene_spec.add_multi_parent_small_objects(
        parsed_llm_data,
        constrained_motif_parent_name_to_id_map
    )

    relevant_small_specs = added_small_objects_spec_container.small_objects

    if not relevant_small_specs:
        logger.warning(f"No small objects were ultimately added to scene_spec for motif {motif.id} from the VLM response")
        return
    else:
        logger.info(f"Added {len(relevant_small_specs)} constrained small ObjectSpecs to scene_spec for motif {motif.id}")

    if not relevant_small_specs:
        logger.warning(f"No small objects found in updated scene spec for motif {motif.id}")
        return
    else:
        logger.info(f"Found {len(relevant_small_specs)} small objects to process for motif {motif.id}")

    analysis_data = {}
    analysis_str = small_obj_session.send_with_validation(
        "populate_surface_motifs",
        {
            "room_type": scene.room_description,
            "small_objects": [obj.to_gpt_dict() for obj in relevant_small_specs],
        },
        lambda response: validate_arrangement_smc(response,
                                                [obj.id for obj in relevant_small_specs],
                                                relevant_small_specs,
                                                enforce_same_layer=True,
                                                enforce_same_surface=True),
        is_json=True,
        verbose=True,
    )
    analysis_data = json.loads(extract_json(analysis_str))

    if not cfg.mode.use_scene_motifs:
        from hsm_core.scene.ablation import create_individual_scene_motifs_with_analysis
        analysis_str = create_individual_scene_motifs_with_analysis(
            [obj.to_dict() for obj in relevant_small_specs], analysis_data
        )
        analysis_data = json.loads(analysis_str)

    # Add height limits from layer data
    for arrangement in analysis_data["arrangements"]:
        if "furniture" in arrangement["composition"]:
            obj_ids = [item["id"] for item in arrangement["composition"]["furniture"]]
            arrangement_objs = [obj for obj in relevant_small_specs if obj.id in obj_ids]

            if arrangement_objs:
                parent_id = arrangement_objs[0].parent_object
                layer_key = arrangement_objs[0].placement_layer

                if parent_id and layer_key:
                    parent_name = None
                    for spec in motif.object_specs:
                        if spec.id == parent_id:
                            parent_name = spec.name
                            break

                    if (layer_data and parent_name and
                        parent_name in layer_data and layer_key in layer_data[parent_name]):
                        layer_info = layer_data[parent_name][layer_key]
                        arrangement["composition"]["height_limit"] = round(layer_info.get("space_above", 0), 4)

    # Process and place small objects
    await process_and_place_small_objects(
        scene, cfg, analysis_data, motif=motif, vis_output_dir=vis_output_dir, object_specs=relevant_small_specs, layer_data=layer_data,
        output_dir=output_dir, model=model, all_object_specs=all_object_specs, solver_fallback=True
    )


def get_remaining_motifs_for_unconstrained_processing(scene, processed_motifs: set[str]) -> set[SceneMotif]:
    """Get motifs that still need unconstrained small object processing."""
    # Calculate remaining motifs, excluding both processed motifs and those with existing small objects
    all_eligible_motifs: set[SceneMotif] = set(filter_motifs_by_types(scene.scene_motifs, [ObjectType.LARGE, ObjectType.WALL]))
    motifs_with_small_objects: set[SceneMotif] = set()

    # Find motifs that already have small objects populated
    for motif in all_eligible_motifs:
        for obj in motif.objects:
            if hasattr(obj, 'child_motifs') and obj.child_motifs:
                motifs_with_small_objects.add(motif)
                break

    processed_motif_objects: set[SceneMotif] = {motif for motif in all_eligible_motifs if motif.id in processed_motifs}
    current_remaining_motifs: set[SceneMotif] = all_eligible_motifs - processed_motif_objects - motifs_with_small_objects

    logger.info(f"Total eligible motifs: {len(all_eligible_motifs)}")
    logger.info(f"Processed motifs (constrained): {len(processed_motifs)} - {list(processed_motifs)}")
    logger.info(f"Motifs with existing small objects: {len(motifs_with_small_objects)} - {[m.id for m in motifs_with_small_objects]}")
    logger.info(f"Remaining {len(current_remaining_motifs)} motifs for unconstrained small objects: {[m.id for m in current_remaining_motifs]}")

    return current_remaining_motifs


async def populate_small_objects(scene, cfg, output_dir: str, vis_output_dir: Path, model=None) -> Optional[Session]:
    """
    Two-phase small object population:
    1. First process motifs that have specifically assigned small objects
    2. Then process remaining motifs with unconstrained small objects
    """
    if not setup_small_objects_population(scene):
        return None

    constrained_small_objects, constrained_parent_motifs = get_constrained_small_objects(scene)
    all_object_specs = build_object_specs_lookup(scene)
    processed_motifs: set[str] = set()

    if constrained_small_objects:
        processed_motifs = await process_constrained_small_objects(
            scene, constrained_small_objects, constrained_parent_motifs,
            all_object_specs, output_dir, cfg, model, vis_output_dir
        )

    return await process_unconstrained_small_objects(
        scene, cfg, processed_motifs, all_object_specs, output_dir, model, vis_output_dir
    )


async def process_unconstrained_small_objects(
    scene, cfg, processed_motifs: set[str],
    all_object_specs: dict[int, tuple[SceneMotif, ObjectSpec]],
    output_dir: str, model, vis_output_dir
) -> Optional[Session]:
    """Process unconstrained small objects in iterations."""
    # Calculate remaining motifs, excluding both processed motifs and those with existing small objects
    remaining_motifs = get_remaining_motifs_for_unconstrained_processing(scene, processed_motifs)

    if "small" not in cfg.mode.extra_types:
        logger.info("Skipping unconstrained small objects from config")
        return None

    ############## Unconstrained small objects with iterations ###################
    max_iterations = cfg.parameters.small_object_generation.max_iterations
    target_saturation = cfg.parameters.small_object_generation.target_occupancy_percent / 100.0
    iteration = 0
    small_obj_session = None

    # Track cumulative occupancy across all surfaces and iterations
    cumulative_occupancy = 0.0
    total_surfaces_processed = 0

    logger.info(f"Starting unconstrained small object iterations (max: {max_iterations}, target saturation: {target_saturation * 100:.1f}%)")


    while iteration < max_iterations:
        logger.info(f"{'='*60}")
        logger.info(f"Unconstrained Small Objects - Iteration {iteration + 1}/{max_iterations}")
        logger.info(f"{'='*60}")

        # Recalculate remaining motifs for each iteration
        remaining_motifs = get_remaining_motifs_for_unconstrained_processing(scene, processed_motifs)

        if not remaining_motifs:
            logger.info(f"No eligible motifs for iteration {iteration + 1}. Stopping iterations")
            break

        logger.info(f"Iteration {iteration + 1}: Processing {len(remaining_motifs)} motifs")
        logger.debug(f"Motif IDs: {[m.id for m in remaining_motifs]}")

        # Process one iteration of unconstrained small objects
        processed_motifs, iteration_session, motifs_processed, iteration_occupancy_data = await process_unconstrained_small_objects_iteration(
            scene, cfg=cfg,
            vis_output_dir=vis_output_dir,
            remaining_motifs=remaining_motifs,
            processed_motifs=processed_motifs,
            all_object_specs=all_object_specs,
            output_dir=output_dir,
            model=model,
            iteration=iteration
        )

        # Update session for return value
        if iteration_session:
            small_obj_session = iteration_session

        # Accumulate occupancy data from this iteration
        if iteration_occupancy_data:
            iteration_occupancy = iteration_occupancy_data.get('average_occupancy', 0.0)
            surfaces_in_iteration = iteration_occupancy_data.get('surfaces_processed', 0)

            if surfaces_in_iteration > 0:
                # Calculate weighted average occupancy across all iterations
                cumulative_occupancy = ((cumulative_occupancy * total_surfaces_processed) +
                                      (iteration_occupancy * surfaces_in_iteration)) / (total_surfaces_processed + surfaces_in_iteration)
                total_surfaces_processed += surfaces_in_iteration

        occupancy_ratio = cumulative_occupancy if total_surfaces_processed > 0 else 0.0
        logger.info(f"Current occupancy ratio: {occupancy_ratio:.1f}")
        logger.info(f"Motifs processed so far: {len(processed_motifs)}/{len(scene.scene_motifs)}")

        # Check if we've reached target saturation or processed no new motifs
        if occupancy_ratio >= target_saturation:
            logger.info(f"Target saturation reached: {occupancy_ratio:.1f} >= {target_saturation:.1f}")
            break
        elif motifs_processed == 0:
            logger.info(f"No new motifs processed in iteration {iteration + 1}. Stopping iterations")
            break
        elif iteration_occupancy_data and iteration_occupancy_data.get('surfaces_processed', 0) == 0:
            logger.info(f"No surfaces processed in iteration {iteration + 1}. Stopping iterations")
            break

        iteration += 1

    logger.info(f"Completed unconstrained small object iterations. Total iterations: {iteration + 1}")
    logger.info(f"Final occupancy ratio: {occupancy_ratio:.1f}")
    logger.info(f"Total surfaces processed: {total_surfaces_processed}")

    return small_obj_session


async def process_unconstrained_small_objects_iteration(
    scene,
    cfg: DictConfig,
    vis_output_dir: Path,
    remaining_motifs: set[SceneMotif],
    processed_motifs: set[str],
    all_object_specs: dict[int, tuple[SceneMotif, ObjectSpec]],
    output_dir: str,
    model=None,
    iteration: int = 0
) -> tuple[set[str], Session, int, dict]:
    """
    Process one iteration of unconstrained small object population.

    Args:
        scene: The scene object
        cfg: Configuration object
        vis_output_dir: Output directory path for visualizations
        remaining_motifs: Set of motifs available for unconstrained processing
        processed_motifs: Set of motif IDs already processed
        all_object_specs: Lookup dictionary for object specifications
        output_dir: Output directory path
        model: ModelManager instance
        iteration: Current iteration number

    Returns:
        Tuple of (updated_processed_motifs, session, motifs_processed_count, occupancy_data)
        where occupancy_data contains:
        - 'average_occupancy': weighted average occupancy ratio from DFS solver
        - 'surfaces_processed': number of surfaces processed in this iteration
    """
    from hsm_core.scene.utils.anchor import find_anchor_object

    motifs_processed_count = 0
    small_obj_session = None

    # Track occupancy data from DFS solver
    total_occupancy = 0.0
    total_surfaces = 0

    logger.info(f"Starting unconstrained small object iteration {iteration}")
    logger.info(f"Processing {len(remaining_motifs)} motifs for unconstrained small objects")

    motifs_skipped_no_specs = 0
    motifs_skipped_no_surfaces = 0
    motifs_processed_successfully = 0

    for motif in remaining_motifs:  # Note: tqdm removed as it's not available in the function scope

        # Skip motifs without object specs or already processed motifs
        if not motif.object_specs:
            logger.info(f"No object specs found for scene motif {motif.id}. Skipping unconstrained small object population for this motif")
            motifs_skipped_no_specs += 1
            continue

        ids: dict[str, str] = {}
        existing_objects, _ = motif.get_objects_by_names()
        for obj in existing_objects:
            try:
                ids[obj.name] = obj.get_mesh_id()
            except Exception as e:
                logger.error(f"Error retrieving mesh id for object {obj.name if 'name' in locals() else obj}: {e}")

        has_support_surfaces = any(check_support_json_exists(id) for id in ids.values())

        if not has_support_surfaces:
            # if all ids not found, skip motif, else store object names and ids for later
            logger.info(f"No support surface data found for any objects in scene motif {motif.id}. Skipping unconstrained small object population for this motif")
            motifs_skipped_no_surfaces += 1
            continue
        # if all ids found, store object names and ids for later
        else:
            object_names: list[str] = list(ids.keys())
            # object_ids: list[str] = list(ids.values())

        logger.info(f"Found {len(ids)} objects with support surface data for scene motif {motif.id}")
        motifs_processed_successfully += 1

        anchor_scene_objects, _ = find_anchor_object(motif, object_names)

        if not anchor_scene_objects:
            logger.info(f"No anchors found in scene motif '{motif.id}'. Skipping unconstrained small object population for this motif")
            continue

        # Extract names from the found anchor SceneObjects for the current motif
        current_motif_anchor_names = [aso.name for aso in anchor_scene_objects if hasattr(aso, 'name') and aso.name]

        # Collect surface data for this motif using the identified anchor_names from the current motif
        layer_data, _, layer_fig = collect_surface_data(
            large_object_names=current_motif_anchor_names,
            motif=motif, # Pass the current motif
            output_dir=output_dir,
            try_ransac=False
        )

        if layer_fig:
            plt.close(layer_fig)

        # Skip when no support-surface data is available – unless these small objects were constrained
        if not layer_data:
            logger.info(f"No surface data collected for anchors in scene motif {motif.id}.")
            continue # Skip this motif if no surface data for its anchors

        # Construct parent_name_to_id_map for the current motif
        current_motif_parent_name_to_id_map = build_parent_name_to_id_map(motif)

        # Remap layer_data keys to use instance names instead of base names
        if layer_data:
            remapped_layer_data = {}
            base_name_to_instance_names = {}

            # Group instance names by base name (batch lookup for efficiency)
            base_name_to_instance_names = group_instances_by_base_name(current_motif_parent_name_to_id_map, scene)

            # Create case-insensitive mapping for layer data keys
            layer_data_case_map = {key.lower(): key for key in layer_data.keys()}
            base_name_case_map = {name.lower(): name for name in base_name_to_instance_names.keys()}

            # Remap layer data using instance names with case-insensitive matching
            for layer_key, layer_info in layer_data.items():
                # Try exact match first
                if layer_key in base_name_to_instance_names:
                    instance_names = base_name_to_instance_names[layer_key]
                    for instance_name in instance_names:
                        remapped_layer_data[instance_name] = layer_info
                # Fall back to case-insensitive match
                elif layer_key.lower() in base_name_case_map:
                    actual_base_name = base_name_case_map[layer_key.lower()]
                    instance_names = base_name_to_instance_names[actual_base_name]
                    for instance_name in instance_names:
                        remapped_layer_data[instance_name] = layer_info

            layer_data = remapped_layer_data

        small_obj_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))

        # Collect information about existing objects on each parent
        existing_objects_info = collect_existing_objects_info(scene, motif, current_motif_parent_name_to_id_map)
        logger.debug(f"Existing objects info for scene motif {motif.id}: {existing_objects_info}")

        # Get additional small objects for this motif with layer-specific placement
        prompt_data = {
            "small_objects": "***Suggest small objects for this scene motif***",
            "motif_description": str(motif.description),
            "large_furniture": str(list(current_motif_parent_name_to_id_map.keys())), # Use instance names instead of base names
            "room_type": scene.room_type,
            "layer_info": json.dumps(clean_layer_info(layer_data))
        }

        # Add existing objects information if available
        if existing_objects_info and any(existing_objects_info.values()):
            prompt_data["existing_objects"] = json.dumps(existing_objects_info)
        else:
            prompt_data["existing_objects"] = "No existing small objects on these surfaces"

        objects_response = small_obj_session.send(
            "small_objects_layered",
            prompt_data,
            is_json=True,
            verbose=True
        )
        layered_response = json.loads(extract_json(objects_response))
        # No filtering needed - the VLM should respect existing objects in the prompt
        filtered_response = layered_response

        all_newly_added_small_objects_for_motif = []

        # Iterate through the instance names from the parent_name_to_id_map
        for instance_name, parent_id in current_motif_parent_name_to_id_map.items():
            parent_spec = scene.scene_spec.get_object_by_id(parent_id)
            if not parent_spec:
                logger.warning(f"Could not find ObjectSpec for parent ID {parent_id} (instance '{instance_name}') in scene motif '{motif.id}'. Skipping this parent")
                continue

            logger.debug(f"Processing parent instance: '{instance_name}' (ID: {parent_id}) within scene motif '{motif.id}'")

            # Check if the VLM response has suggestions for this specific instance name
            llm_response_key_for_parent = None
            for key_from_llm in filtered_response.keys():
                if key_from_llm.lower() == instance_name.lower():
                    llm_response_key_for_parent = key_from_llm
                    break

            if llm_response_key_for_parent and isinstance(filtered_response.get(llm_response_key_for_parent), dict):
                sub_response_for_parent = {llm_response_key_for_parent: filtered_response[llm_response_key_for_parent]}

                # scene.scene_spec is mutated by add_layered_objects_from_response (via scene.add_objects)
                # The returned 'added_spec' contains only the new ObjectSpecs from this specific call.
                added_spec_for_this_parent = scene.scene_spec.add_layered_objects_from_response(
                    sub_response_for_parent,
                    parent_id, # This is the ID of the ObjectSpec within the current motif
                    instance_name # This is the instance name for this specific parent
                )

                if added_spec_for_this_parent and added_spec_for_this_parent.small_objects:
                    all_newly_added_small_objects_for_motif.extend(added_spec_for_this_parent.small_objects)
                else:
                    logger.debug(f"No new small objects added by add_layered_objects_from_response for parent instance '{instance_name}' (ID: {parent_id})")
            else:
                logger.debug(f"No specific suggestions found in VLM response for parent instance '{instance_name}' (ID: {parent_id}) in motif '{motif.id}', or response format invalid. VLM response keys: {list(filtered_response.keys())}")

        # After iterating all parent instances for the current motif:
        relevant_small_specs = all_newly_added_small_objects_for_motif

        if not relevant_small_specs:
            logger.info(f"No small objects found in updated scene spec for motif {motif.id}")
            continue

        logger.info(f"Found {len(relevant_small_specs)} small objects to process for motif {motif.id}")

        # group small objects to motifs by parent_object
        analysis_data = {}
        analysis_str = small_obj_session.send_with_validation(
            "populate_surface_motifs",
            {
                "room_type": scene.room_description,
                "small_objects": [obj.to_gpt_dict() for obj in relevant_small_specs],
            },
            lambda response: validate_arrangement_smc(response,
                                                    [obj.id for obj in relevant_small_specs],
                                                    relevant_small_specs,
                                                    enforce_same_layer=True,
                                                    enforce_same_surface=True),
            is_json=True,
            verbose=True,
        )
        analysis_data = json.loads(extract_json(analysis_str))

        if not cfg.mode.use_scene_motifs:
            from hsm_core.scene.ablation import create_individual_scene_motifs_with_analysis
            analysis_str = create_individual_scene_motifs_with_analysis([obj.to_dict() for obj in relevant_small_specs], analysis_data)
            analysis_data = json.loads(analysis_str)

        # Add a height limit to the analysis for each arrangement from the layer_data
        for arrangement in analysis_data["arrangements"]:
            # Extract parent object and layer information from the objects in this arrangement
            if "furniture" in arrangement["composition"]:
                obj_ids = [item["id"] for item in arrangement["composition"]["furniture"]]
                # Find the corresponding objects
                arrangement_objs = [obj for obj in relevant_small_specs if obj.id in obj_ids]

                if arrangement_objs:
                    # Get parent and layer info from the first object (assuming all have same parent/layer)
                    parent_id = arrangement_objs[0].parent_object
                    layer_key = arrangement_objs[0].placement_layer

                    if parent_id and layer_key:
                        # Get parent name
                        parent_name = None
                        for spec in motif.object_specs:
                            if spec.id == parent_id:
                                parent_name = spec.name
                                break

                        # Add height limit from layer data
                        if layer_data and parent_name and parent_name in layer_data and layer_key in layer_data[parent_name]:
                            layer_info = layer_data[parent_name][layer_key]
                            arrangement["composition"]["height_limit"] = round(layer_info.get("space_above", 0), 4)

        try:
            # Process and place these small objects
            motif_occupancy = await process_and_place_small_objects(scene, cfg, analysis_data, motif, vis_output_dir, relevant_small_specs, layer_data, output_dir, model, all_object_specs)

            # Accumulate occupancy data from this motif
            if motif_occupancy and isinstance(motif_occupancy, dict):
                total_occupancy += motif_occupancy.get('total_occupancy', 0.0)
                total_surfaces += motif_occupancy.get('surfaces_processed', 0)

            processed_motifs.add(motif.id)
            motifs_processed_count += 1
        except Exception as e:
            logger.error(f"Skipping scene motif {motif.id} small objects unconstrained population due to error: {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info(f"Completed unconstrained small object iteration {iteration}. Processed {motifs_processed_count} scene motifs")
    
    # Iteration summary 
    # - Total motifs input: {len(remaining_motifs)}
    # - Motifs processed successfully: {motifs_processed_successfully}
    # - Motifs skipped (no specs): {motifs_skipped_no_specs}
    # - Motifs skipped (no surfaces): {motifs_skipped_no_surfaces}
    # - Total occupancy accumulated: {total_occupancy}
    # - Total surfaces processed: {total_surfaces}

    iteration_occupancy_data = {
        'average_occupancy': total_occupancy / total_surfaces if total_surfaces > 0 else 0.0,
        'surfaces_processed': total_surfaces,
        'total_occupancy': total_occupancy
    }

    return processed_motifs, small_obj_session, motifs_processed_count, iteration_occupancy_data


async def process_and_place_small_objects(scene, cfg, analysis, motif: SceneMotif, vis_output_dir: Path,
                                       object_specs: list[ObjectSpec], layer_data: dict, output_dir: str,
                                       model=None,
                                       all_object_specs: dict = {},
                                       solver_fallback=False):
    """Helper method to process and place small objects for a motif"""
    if not analysis:
        logger.warning(f"No analysis provided for scene motif {motif.id}")
        return {}

    try:
        layout_data = analysis
        motif_occupancy_data = {
            'average_occupancy': 0.0,
            'surfaces_processed': 0,
            'total_occupancy': 0.0
        }

        # Create support surface constraints for small objects
        support_surface_constraints = create_support_surface_constraints(
            scene, object_specs, layer_data, motif
        )

        # Generate initial small motifs from layout data
        initial_small_motifs, _ = await process_scene_motifs(
            object_specs,
            layout_data,
            output_dir=output_dir,
            room_description=scene.room_description,
            model=model,
            object_type=ObjectType.SMALL,
            support_surface_constraints=support_surface_constraints
        )

        logger.info(f"Generated {len(initial_small_motifs)} initial scene motifs for {motif.id}: {[m.id for m in initial_small_motifs]}")

        if not initial_small_motifs:
            logger.warning(f"No small motifs generated for scene motif {motif.id}")
            return

        # Deduplicate small motifs by ID to prevent duplicates
        deduplicated_motifs = {}
        for sm in initial_small_motifs:
            if sm.id not in deduplicated_motifs:
                deduplicated_motifs[sm.id] = sm

        initial_small_motifs = list(deduplicated_motifs.values())
        parent_to_motifs = {}
        processed_small_motif_ids = set()

        for small_motif in initial_small_motifs:
            # Skip if this small motif has already been processed
            if small_motif.id in processed_small_motif_ids:
                continue

            processed_small_motif_ids.add(small_motif.id)

            # Find the parent_id for this small motif
            parent_ids = set()
            for spec in small_motif.object_specs:
                if spec.parent_object:
                    parent_ids.add(spec.parent_object)

            # Add this motif to each parent's list (only once per parent)
            for parent_id in parent_ids:
                if parent_id not in parent_to_motifs:
                    parent_to_motifs[parent_id] = []
                # Only add if not already in this parent's list
                if not any(m.id == small_motif.id for m in parent_to_motifs[parent_id]):
                    parent_to_motifs[parent_id].append(small_motif)

        logger.info(f"Grouped small motifs by parent: {[(pid, [m.id for m in motifs]) for pid, motifs in parent_to_motifs.items()]}")

        # Process each parent object using the PASSED dictionary
        for parent_id, relevant_motifs in parent_to_motifs.items():
            # Check if the parent_id corresponds to an object within the `motif` argument.
            # The `motif` argument is the primary large object motif context (e.g., floating_wall_shelf).
            parent_spec_in_current_motif = None
            for spec_in_main_motif in motif.object_specs: # `motif` is the one passed to _process_and_place_small_objects
                if spec_in_main_motif.id == parent_id:
                    parent_spec_in_current_motif = spec_in_main_motif
                    break

            target_motif: SceneMotif
            parent_spec: ObjectSpec

            if parent_spec_in_current_motif:
                # The parent object is within the current processing motif. Use this motif directly.
                target_motif = motif
                parent_spec = parent_spec_in_current_motif
            else:
                # Fallback to the global map if the parent_id isn't directly in the current `motif`'s specs.
                # This might happen if parent_id refers to an object in a different motif.
                if not scene.scene_spec:
                    logger.warning(f"SceneSpec not available for global lookup of parent ID {parent_id}")
                    continue
                parent_info_from_map = all_object_specs.get(parent_id)
                if not parent_info_from_map:
                    logger.warning(f"Parent object ID {parent_id} not found in current motif '{motif.id}' or global lookup")
                    continue
                target_motif, parent_spec = parent_info_from_map

            # Get layout suggestions first
            layout_suggestions, layer_data, layer_fig = populate_furniture(
                large_object_names=[parent_spec.name],
                room_desc=scene.room_description,
                generated_small_motif=relevant_motifs,
                motif=target_motif
            )

            # Skip when no support-surface data is available – unless these small objects were explicitly constrained
            if not layer_data:
                logger.info(f"No layer data found for {parent_spec.name}")
                continue

            # Get the constrained layout for this parent
            parent_occupancy_data, constrained_layout = optimize_small_objects(
                cfg,
                fig=target_motif.fig,
                layout_suggestions=layout_suggestions,
                output_path=os.path.join(vis_output_dir, f"{target_motif.id}", "solver"),
                layer_fig=layer_fig,
                layer_data=layer_data,
                small_motifs=relevant_motifs,
                fallback=solver_fallback,
                scene=scene
            )

            # Accumulate occupancy data from this parent
            if parent_occupancy_data:
                motif_occupancy_data['total_occupancy'] += parent_occupancy_data.get('total_occupancy', 0.0)
                motif_occupancy_data['surfaces_processed'] += parent_occupancy_data.get('surfaces_processed', 0)

            updated_motifs = update_small_motifs_from_constrained_layout(
                constrained_layout, relevant_motifs, target_motif, layer_data
            )

            logger.info(f"Got {len(updated_motifs)} updated scene motifs for {parent_spec.name}")

            # Find the actual parent object and add child motifs to it
            parent_object = target_motif.get_object(parent_spec.name)
            if parent_object:
                if updated_motifs:
                    logger.info(f"Adding {len(updated_motifs)} scene motifs to parent {parent_object.name}")
                    parent_object.child_motifs.extend(updated_motifs)
                    scene.add_motifs(updated_motifs)
            else:
                logger.warning(f"Could not find parent object {parent_spec.name} in scene motif {motif.id}")

            # Run spatial optimization for the small objects after placement
            if updated_motifs and cfg.mode.get('enable_spatial_optimization', False):
                try:
                    from hsm_core.scene.processing.processing_helpers import run_spatial_optimization_for_stage, filter_motifs_needing_optimization
                    motifs_needing_optimization = filter_motifs_needing_optimization(updated_motifs)
                    if motifs_needing_optimization:
                        run_spatial_optimization_for_stage(
                            scene=scene,
                            cfg=cfg,
                            current_stage_motifs=motifs_needing_optimization,
                            object_type=ObjectType.SMALL,
                            output_dir=Path(output_dir),
                            stage_name=f"parent_{parent_spec.name}"
                        )
                except Exception as e:
                    logger.warning(f"Spatial optimization failed for small objects on {parent_spec.name}: {e}")
                    logger.error(traceback.format_exc())

        # Calculate average occupancy for this motif
        if motif_occupancy_data['surfaces_processed'] > 0:
            motif_occupancy_data['average_occupancy'] = (
                motif_occupancy_data['total_occupancy'] / motif_occupancy_data['surfaces_processed']
            )

        return motif_occupancy_data

    except Exception as e:
        logger.error(f"Error processing small objects for scene motif {motif.id}: {e}")
        logger.error(traceback.format_exc())


def collect_existing_objects_info(scene, motif: SceneMotif, parent_name_to_id_map: dict) -> dict:
    """
    Collect information about existing small objects on each parent object.

    Args:
        scene: The scene object
        motif: The motif containing parent objects
        parent_name_to_id_map: Mapping from instance names to object IDs

    Returns:
        dict: Information about existing objects on each parent
    """
    existing_info = {}
    for instance_name, parent_id in parent_name_to_id_map.items():
        # Find the parent object in the motif
        parent_obj = None

        # Try by ID if objects have ID attribute
        for obj in motif.objects:
            if hasattr(obj, 'id') and obj.id == parent_id:
                parent_obj = obj
                break

        # Try by name matching (instance_name might be like "nightstand_1")
        if not parent_obj:
            for obj in motif.objects:
                if hasattr(obj, 'name'):
                    # Check if instance_name contains the object name or vice versa
                    if (instance_name in obj.name or
                        obj.name in instance_name or
                        instance_name.split('_')[0] == obj.name):
                        parent_obj = obj
                        break

        if not parent_obj:
            continue

        # Check child_motifs for existing small objects
        existing_objects = []
        if hasattr(parent_obj, 'child_motifs') and parent_obj.child_motifs:
            for child_motif in parent_obj.child_motifs:
                if hasattr(child_motif, 'object_specs') and child_motif.object_specs:
                    for obj_spec in child_motif.object_specs:
                        existing_objects.append({
                            'name': obj_spec.name,
                            'type': obj_spec.object_type.value if hasattr(obj_spec, 'object_type') else 'unknown'
                        })

        if existing_objects:
            existing_info[instance_name] = existing_objects

    return existing_info


def create_support_surface_constraints(
    scene,
    small_objects: list[ObjectSpec],
    layer_data: dict,
    parent_motif: SceneMotif
) -> dict[str, dict]:
    """
    Create support surface constraints for small objects based on their parent objects and layer data.

    Args:
        scene: The scene object
        small_objects: List of small object specs that need support surface constraints
        layer_data: Dictionary containing layer information for parent objects
        parent_motif: Parent motif containing the parent objects

    Returns:
        dict[str, dict]: Mapping of object labels to their support surface constraints
    """
    constraints = {}

    # Create lookup for parent objects in the motif
    parent_objects = {}
    for obj in parent_motif.objects:
        if hasattr(obj, 'name'):
            parent_objects[obj.name] = obj

    for small_obj in small_objects:
        if not hasattr(small_obj, 'name') or not hasattr(small_obj, 'parent_object'):
            continue

        # Find parent object info
        parent_id = small_obj.parent_object
        parent_spec = None
        for spec in parent_motif.object_specs:
            if spec.id == parent_id:
                parent_spec = spec
                break

        if not parent_spec:
            continue

        parent_name = parent_spec.name
        placement_layer = getattr(small_obj, 'placement_layer', None)
        placement_surface = getattr(small_obj, 'placement_surface', None)

        # Get layer information for this parent and layer
        if layer_data and parent_name in layer_data and placement_layer in layer_data[parent_name]:
            layer_info = layer_data[parent_name][placement_layer]

            # Build constraint dict
            surface_constraints = {}

            if placement_surface is not None and 'surfaces' in layer_info:
                # Surface-level constraints
                for surface in layer_info['surfaces']:
                    if surface.get('surface_id') == placement_surface:
                        w = surface.get('width', 1.0)
                        d = surface.get('depth', 1.0)
                        area = surface.get('area', w * d)
                        bounds = surface.get('bounds', {'width': w, 'depth': d})
                        surface_constraints = {
                            'available_area': area,
                            'bounds': bounds,
                            'max_height': layer_info.get('space_above', 0.5),
                            'parent_name': parent_name,
                            'layer': placement_layer,
                            'surface_id': placement_surface,
                        }
                        break
            else:
                # Layer-level fallback (no explicit surface)
                w = 0.8
                d = 0.8
                area = w * d
                surface_constraints = {
                    'available_area': area,
                    'bounds': {'width': w, 'depth': d},
                    'max_height': layer_info.get('space_above', 0.5),
                    'parent_name': parent_name,
                    'layer': placement_layer,
                }

            if surface_constraints:
                constraints[small_obj.name.lower()] = surface_constraints
                logger.debug(f"Created support surface constraints for {small_obj.name}: {surface_constraints}")

    return constraints


def check_support_json_exists(mesh_id: str) -> bool:
    """Check if support surface JSON exists for a given mesh ID."""
    from hsm_core.support_region.loader import check_support_json_exists as check_exists
    return check_exists(mesh_id)


