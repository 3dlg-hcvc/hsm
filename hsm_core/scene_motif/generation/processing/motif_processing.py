"""
Motif processing functionality for compositional scene generation.

This module inference individual motifs with visual validation and optimization.
"""

import json
import logging
import traceback
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

import hsm_core.vlm.gpt as gpt
from ..decomposition import decompose_motif_with_session
from hsm_core.scene_motif import MotifHierarchy
from hsm_core.utils import get_logger
from hsm_core.utils.logging import get_motif_logger
from .processors import build_arrangement_from_json
from ...utils import is_sm_exceeds_support_region
from ...generation.llm import send_llm_with_images_async
from hsm_core.scene_motif.utils.motif_visualize import generate_all_motif_views
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.config import PROMPT_DIR
from hsm_core.utils.plot_utils import combine_figures
from ...utils import persist_motif_arrangement

if TYPE_CHECKING:
    from hsm_core.scene.core.motif import SceneMotif

VALIDATE_THRESHOLD = 0.4  # Threshold for validating arrangement
MAKE_TIGHT_THRESHOLD = 0.5  # Threshold for making objects tight

async def process_motif_with_visual_validation(
    motif: "SceneMotif",
    furniture_map: Dict[str, List],
    output_dir: Path,
    save_prefix: str,
    optimize: bool,
    force_make_tight: bool,
    skip_visual_validation: bool,
    log_to_terminal: bool,
    support_surface_constraints: Optional[Dict[str, Dict]],
    first_decompose_result: Optional[Tuple[Optional[str], Optional[Dict]]] = None,
    max_attempts: int = 3
) -> Optional["SceneMotif"]:
    """Generate a scene motif with visual feedback"""
    save_name = f"{save_prefix}_{motif.id}" if save_prefix else motif.id
    motif_output_dir = output_dir / save_name
    motif_output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_motif_logger('scene_motif.processing', motif.id, motif_output_dir)
    logger.info(f"Starting motif processing with pre-computed decomposition")
    std_logger = logging.getLogger('hsm_core.scene_motif.processing')

    decompose_session = gpt.Session(
        str(PROMPT_DIR / "sm_prompts_decompose.yaml"),
        prompt_info={"MOTIF_DEFINITIONS": yaml.safe_load(open(PROMPT_DIR / "motif_definitions.yaml"))["motifs"]}
    )

    for attempt in range(max_attempts):
        logger.info(f"Attempt {attempt + 1}/{max_attempts} for motif {motif.id}")
        
        try:
            # Step 1: Get arrangement JSON (use pre-computed for first attempt, re-decompose for subsequent)
            if attempt == 0 and first_decompose_result is not None:
                # Use pre-computed decomposition result for first attempt
                if isinstance(first_decompose_result, Exception):
                    logger.warning(f"Decomposition failed with exception: {first_decompose_result}")
                    arrangement_json, validate_response = None, None
                elif (first_decompose_result is not None and 
                        isinstance(first_decompose_result, tuple) and 
                        len(first_decompose_result) == 2):
                    arrangement_json, validate_response = first_decompose_result
                    if arrangement_json is not None:
                        logger.info(f"Using decomposition from first attempt")
                else:
                    logger.warning(f"Invalid first decomposition result format: {type(first_decompose_result)}")
                    arrangement_json, validate_response = None, None
            else:
                # Re-decompose with accumulated feedback for subsequent attempts
                if attempt > 0:
                    decompose_session.add_feedback(f"Previous visual validation failed. You must change your decomposition approach.")
                
                logger.info(f"Re-decomposing for attempt {attempt + 1}")
                arrangement_json, validate_response = await decompose_motif_with_session(
                    motif, std_logger, decompose_session, max_decompose_attempts=2
                )
            
            if arrangement_json is None:
                logger.info(f"Decomposition failed on attempt {attempt + 1}")
                continue
            
            # Step 2: Process arrangement
            retrieved_furniture = furniture_map[motif.id]
            logger.info(f"Processing arrangement for motif: {motif.id}")

            inference_session = gpt.Session(str(PROMPT_DIR / "sm_prompts_inference.yaml"))
            # Reset context to prevent contamination from previous attempts
            success, final_arrangement, main_call, sub_arrangements, sub_arr_objs = await build_arrangement_from_json(
                inference_session, arrangement_json, motif.object_specs, retrieved_furniture)
            
            if not success:
                logger.info(f"Arrangement processing failed on attempt {attempt + 1}")
                decompose_session.add_feedback(f"The generated arrangement ({main_call}) could not be executed.")
                continue

            # Step 3: Apply spatial optimization before visual validation
            make_tight = force_make_tight
            if not make_tight:
                make_tight_response = inference_session.send_with_validation("make_tight", {"description": motif.description}, is_json=True)
                make_tight_response = json.loads(make_tight_response)
                make_tight = make_tight_response.get("touch_probability", 0) >= MAKE_TIGHT_THRESHOLD
                
                if support_surface_constraints:
                    logger.info(f"Checking if arrangement exceeds support surface constraints: {support_surface_constraints}")
                    if is_sm_exceeds_support_region(final_arrangement, support_surface_constraints):
                        logger.info("Arrangement exceeds support surface constraints, enabling make_tight")
                        make_tight = True

            hierarchical_success = False
            if optimize and arrangement_json:
                try:
                    from hsm_core.scene_motif.spatial import optimize_sm

                    # Reuse carried hierarchy when available; otherwise build from JSON
                    hierarchy = getattr(final_arrangement, "_hierarchy", None)
                    if hierarchy is None:
                        arrangement_json_data = json.loads(gpt.extract_json(arrangement_json))
                        hierarchy = MotifHierarchy()
                        hierarchy.build_hierarchy(arrangement_json_data, sub_arrangements)

                    for (sub_type, sub_call), sub_arr in zip(sub_arrangements, sub_arr_objs):
                        nodes = hierarchy.get_nodes_by_type(sub_type)
                        for node in nodes:
                            if node.arrangement_call == sub_call:
                                hierarchy.set_arrangement(node, sub_arr)
                                break

                    if hierarchy and hierarchy.root:
                        hierarchy.set_arrangement(hierarchy.root, final_arrangement)

                    final_arrangement = optimize_sm(
                        final_arrangement,
                        hierarchy=hierarchy,
                        make_tight=make_tight,
                    )
                    logger.info(f"Hierarchical spatial optimization completed successfully")
                    hierarchical_success = True

                except Exception as hierarchy_e:
                    logger.error(f"Hierarchical optimization failed: {hierarchy_e}")
                    logger.warning(f"Falling back to standard optimization...")
                    hierarchical_success = False

            if optimize and not hierarchical_success:
                logger.info(f"Applying standard spatial optimization...")
                final_arrangement = optimize_sm(
                    final_arrangement,
                    make_tight=make_tight
                )
                logger.info(f"Standard spatial optimization completed successfully")
            elif not optimize:
                logger.info(f"Optimization skipped as requested (optimize=False)")

            # Step 4: Generate visualization for visual validation
            individual_figs_dict = generate_all_motif_views(
                scene=final_arrangement.to_scene(),
                output_path=str(motif_output_dir),
                verbose=False,
                name=f"{motif.id}_attempt_{attempt + 1}",
                global_scene_transform=np.array([[1,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,1]])
            )

            combined_fig = None
            if individual_figs_dict:
                figures_to_combine = [fig for fig in individual_figs_dict.values() if fig is not None]
                if figures_to_combine:
                    combined_fig_path = motif_output_dir / f"{save_name}_attempt_{attempt + 1}_views.png"
                    combined_fig = combine_figures(
                        figures=figures_to_combine,
                        num_cols=len(figures_to_combine),
                        figsize=(5 * len(figures_to_combine), 5),
                        output_path=str(combined_fig_path)
                    )
                    logger.info(f"Generated visualization for attempt {attempt + 1}")

            # Step 5: Visual validation
            logger.info(f"Starting visual validation for motif {motif.id} (attempt {attempt + 1})")
            visual_validation_passed = True
            visual_feedback = ""
            
            if not skip_visual_validation and combined_fig is not None:
                logger.info(f"Visualization generated successfully")
                validate_session = gpt.Session(str(PROMPT_DIR / "sm_prompts_inference.yaml"))
                try:
                    logger.info(f"Sending scene motif to VLM for visual validation...")
                    logger.info(f"Validating scene motif description: '{motif.description}'")
                    
                    validation_result = await send_llm_with_images_async(
                        validate_session, "validate",
                        {"description": motif.description},
                        images=combined_fig, is_json=True, verbose=True, image_detail='auto'
                    )
                    
                    validation_response = json.loads(validation_result)
                    if isinstance(validation_response, dict):
                        validation_score = validation_response.get("correct", 0)
                        valid_arrangement = validation_score >= VALIDATE_THRESHOLD
                        logger.info(f"Visual validation score: {validation_score:.2f} (threshold: {VALIDATE_THRESHOLD})")
                        logger.info(f"Visual validation result: {'PASSED' if valid_arrangement else 'FAILED'}")
                        
                        if not valid_arrangement:
                            visual_validation_passed = False
                            visual_feedback = validation_response.get('feedback', 'Visual validation failed without specific feedback')
                            logger.info(f"Visual validation failed on attempt {attempt + 1}: {visual_feedback}")
                        else:
                            logger.info(f"Visual validation passed on attempt {attempt + 1}")
                            
                except Exception as e:
                    logger.info(f"Visual validation error on attempt {attempt + 1}: {e}")
                    logger.error(f"Continuing without visual validation due to error")
            else:
                logger.info(f"No visualization available for validation") if not skip_visual_validation else logger.info(f"Skipped visual validation")

            # Step 6: Handle validation result
            if visual_validation_passed:
                logger.info(f"Completing processing for motif {motif.id} (attempt {attempt + 1})")
                
                motif = await persist_motif_arrangement(
                    motif,
                    final_arrangement=final_arrangement,
                    output_dir=motif_output_dir,
                    arrangement_id=motif.id,
                    furniture_specs=motif.object_specs,
                    arrangement_json=arrangement_json,
                    validate_response=validate_response,
                    main_call=main_call,
                    sub_arrangements=sub_arrangements,
                    optimize=optimize,
                    make_tight=make_tight,
                )
                
                success_msg = (f"Successfully processed motif {motif.id} on attempt {attempt + 1}. "
                                f"Saved to {motif.glb_path}")
                logger.info(success_msg)
                
                return motif
            else:
                decompose_session.add_feedback(f"VISUAL VALIDATION FAILED: {visual_feedback}")
                
        except Exception as e:
            logger.info(f"Exception on attempt {attempt + 1}: {str(e)}")
            logger.info(f"Traceback: {traceback.format_exc()}")
            continue
        
    logger.info(f"All attempts failed for motif {motif.id}")
    return None

async def process_single_object_motifs(
    scene_motif_specs: List["SceneMotif"],
    output_dir: Path,
    model,
    save_prefix: str,
    mesh_overrides: Optional[Dict[str, str]],
    object_type: ObjectType,
    log_to_terminal: bool,
    support_surface_constraints: Optional[Dict[str, Dict]]
) -> Tuple[List["SceneMotif"], List["SceneMotif"], Dict[str, List], List]:
    """Separates single from multi-motifs, processing single ones immediately."""
    from .processors import process_single_furniture_arrangement

    global_logger = get_logger('scene_motif.processing.multi')

    processed_single_motifs: List["SceneMotif"] = []
    multi_object_motifs: List["SceneMotif"] = []
    all_furniture = []
    furniture_map = {}

    global_logger.info(f"Processing {len(scene_motif_specs)} motifs...")
    
    for idx, motif in enumerate(scene_motif_specs):
        global_logger.info(f"Processing motif {idx + 1}/{len(scene_motif_specs)}: {motif.id}")

        if not motif.object_specs:
            global_logger.warning(f"Motif {motif.id} has no furniture specs, skipping")
            continue

        global_logger.info(f"  - Object specs: {len(motif.object_specs)}")
        for spec in motif.object_specs:
            global_logger.info(f"    * {spec.name} (amount: {spec.amount})")
        
        if len(motif.object_specs) == 1 and motif.object_specs[0].amount == 1:
            global_logger.info(f"  - Identified as single-object motif, processing immediately...")
            try:
                processed_motif = await process_single_furniture_arrangement(
                    motif, output_dir, save_prefix, model, mesh_overrides,
                    object_type, log_to_terminal, support_surface_constraints
                )
                if processed_motif:
                    global_logger.info(f"  - Successfully processed single-object motif: {motif.id}")
                    processed_single_motifs.append(processed_motif)
                else:
                    global_logger.error(f"  - Failed to process single-object motif: {motif.id}")
            except Exception as e:
                global_logger.error(f"Error processing single furniture motif {motif.id}: {str(e)}")
                global_logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            global_logger.info(f"  - Identified as multi-object motif, adding to queue...")
            multi_object_motifs.append(motif)
            arrangement_furniture = []
            for spec in motif.object_specs:
                for _ in range(spec.amount):
                    obj = spec.to_obj()
                    obj.label = spec.name.lower()
                    arrangement_furniture.append(obj)
            all_furniture.extend(arrangement_furniture)
            furniture_map[motif.id] = arrangement_furniture
    
    global_logger.debug(f"Motif separation complete:")
    global_logger.debug(f"  - Single-object motifs processed: {len(processed_single_motifs)}")
    global_logger.debug(f"  - Multi-object motifs queued: {len(multi_object_motifs)}")
    global_logger.debug(f"  - Total furniture objects: {len(all_furniture)}")
            
    return multi_object_motifs, processed_single_motifs, furniture_map, all_furniture 