import json
import traceback
from copy import deepcopy, copy
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, TYPE_CHECKING
import numpy as np

import hsm_core.vlm.gpt as gpt
from hsm_core.retrieval.core.adaptive_retrieval import retrieve_adaptive
from ...utils.library import load
from ...core.arrangement import Arrangement
from ...core.obj import Obj
from ..llm import generate_arrangement_code
from ...utils import extract_objects, persist_motif_arrangement, create_furniture_lookup, assign_mesh_to_object
from hsm_core.utils import get_logger
from hsm_core.utils.logging import get_motif_logger
from ...programs.interpretor import execute_with_context
from hsm_core.scene_motif.utils.motif_visualize import visualize_scene_motif
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.specifications.object_spec import ObjectSpec

if TYPE_CHECKING:
    from hsm_core.scene.core.motif import SceneMotif

global_logger = get_logger('scene_motif.generation.processing')


async def process_single_furniture_arrangement(
    motif: "SceneMotif",
    output_dir: Path,
    save_prefix: str,
    model,
    mesh_overrides: Optional[Dict[str, str]] = None,
    object_type: ObjectType = ObjectType.UNDEFINED,
    log_to_terminal: bool = True,
    support_surface_constraints: Optional[Dict[str, Dict]] = None
) -> Optional["SceneMotif"]:
    """Process an arrangement with a single piece of furniture."""
    save_name = f"{save_prefix}_{motif.id}" if save_prefix else motif.id
    motif_output_dir = output_dir / save_name
    motif_output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_motif_logger('scene_motif.processing', motif.id, motif_output_dir)
    try:
        spec = motif.object_specs[0]
        single_obj = spec.to_obj()
        single_obj.label = spec.name.lower()

        logger.info(f"Processing single furniture: {spec.name}")
        logger.debug(f"Object label: {single_obj.label}")

        try:
            await retrieve_adaptive(
                motif_description="N/A",
                objs=[single_obj],
                same_per_label=True,
                avoid_used=False,
                randomize=False,
                use_top_k=5,
                model=model,
                object_type=object_type,
                max_height=motif.height_limit,
                support_surface_constraints=support_surface_constraints or {}
            )
            logger.info(f"Successfully retrieved mesh for {spec.name}")
        except Exception as e:
            logger.error(f"Failed to retrieve mesh for {spec.name}: {e}")
            return None

        if not hasattr(single_obj, 'mesh') or single_obj.mesh is None:
            logger.error(f"No mesh assigned to {spec.name} after retrieval")
            return None

        final_arrangement = Arrangement([single_obj], motif.description, "Single object motif")

        # Generate and save visualization for single view
        try:
            motif.fig, _, _, _, _, _ = visualize_scene_motif(
                scene=final_arrangement.to_scene(),
                output_path=str(motif_output_dir),
                verbose=False,
                name=motif.id,
                transform_matrix=np.array([[1,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,1]])
            )
            logger.info(f"Successfully generated visualization for {spec.name}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization for {spec.name}: {e}")

        try:
            motif = await persist_motif_arrangement(
                motif,
                final_arrangement=final_arrangement,
                output_dir=motif_output_dir,
                arrangement_id=motif.id,
                furniture_specs=motif.object_specs,
                main_call=final_arrangement.function_call,
                optimize=False,
                make_tight=False,
            )
            logger.info(f"Successfully persisted arrangement for {spec.name}")
        except Exception as e:
            logger.error(f"Failed to persist arrangement for {spec.name}: {e}")
            return None

        motif.arrangement = final_arrangement
        logger.info(f"Successfully processed single furniture arrangement for {spec.name}")
        return motif

    except Exception as e:
        logger.error(f"Error in process_single_furniture_arrangement: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


async def build_arrangement_from_json(
    inference_session,
    arrangement_json: str,
    furniture_specs: List[ObjectSpec],
    retrieved_furniture: List[Obj],
) -> Tuple[bool, Any, Any, Any, Any]:
    """Build a complete Arrangement object from JSON spec.
        1. Generate function call
        2. Build MotifHierarchy from arrangement_json
        3. Execute code bottom-up using execute_with_context
        4. Assign meshes and create final Arrangement object

    Args:
        inference_session: VLM session for code generation
        arrangement_json: JSON specification of the arrangement
        furniture_specs: Object specifications for mesh assignment
        retrieved_furniture: Retrieved furniture objects with meshes

    Returns:
        Tuple of (success, final_arrangement, main_call, sub_arrangements, execute_results)
    """
    try:
        global_logger.info("Phase 1: Generating function call...")
        furniture_lookup = create_furniture_lookup(furniture_specs, retrieved_furniture)

        main_call, sub_arrangements = await generate_arrangement_code(
            inference_session,
            arrangement_json,
            retrieved_furniture,
            furniture_lookup=furniture_lookup
        )

        global_logger.debug(f"Received main_call: {main_call[:100] if main_call else 'None/Empty'}...")
        global_logger.debug(f"Received {len(sub_arrangements)} sub_arrangements")

        global_logger.info("Phase 2: Building MotifHierarchy...")
        from hsm_core.scene_motif import MotifHierarchy
        hierarchy = MotifHierarchy()
        arrangement_json_data = json.loads(gpt.extract_json(arrangement_json))
        # Convert 3-tuples to 2-tuples for hierarchy building
        hierarchy_arrangements = [(motif_type, code_string) for motif_type, code_string, _ in sub_arrangements]
        hierarchy.build_hierarchy(arrangement_json_data, hierarchy_arrangements)

        global_logger.info("Phase 3: Executing arrangements bottom-up...")
        arrangement_lookup = {}
        execute_results = []

        # Execute sub-arrangements first
        for idx, (sub_type, sub_call, _) in enumerate(sub_arrangements):
            global_logger.debug(f"Executing sub-arrangement {idx}: {sub_type}")
            sub_meta_program = load(sub_type, is_meta=True)[0]
            sub_result, _ = execute_with_context(sub_meta_program, sub_call, execute_results)

            sub_arrangement = Arrangement(sub_result["objs"], f"{sub_type} sub-arrangement", sub_call)

            # Process meshes for sub-arrangement objects
            for obj in sub_arrangement.objs:
                if isinstance(obj, Obj):
                    assigned = assign_mesh_to_object(obj, furniture_lookup, global_logger)
                    global_logger.info(f"      Assign mesh: label='{obj.label}', assigned={assigned}, mesh_path='{getattr(obj, 'mesh_path', None)}'")

                    # Handle sub-arrangement references
                    if obj.label.startswith('sub_arrangements['):
                        ref_idx = int(obj.label.split('[')[1].split(']')[0])
                        if f"sub_arrangements[{ref_idx}]" in arrangement_lookup:
                            obj._referenced_arrangement = arrangement_lookup[obj.label]

            arrangement_lookup[f"sub_arrangements[{idx}]"] = sub_arrangement
            execute_results.append(sub_arrangement)

            # Update hierarchy with executed arrangement
            nodes = hierarchy.get_nodes_by_type(sub_type)
            if nodes:
                node = next((n for n in nodes if n.depth == 1), nodes[0])  # Assume depth 1 for sub-arrangements
                hierarchy.set_arrangement(node, sub_arrangement)

        global_logger.debug("Executing main arrangement")
        main_arrangement_data = json.loads(gpt.extract_json(arrangement_json))
        main_meta_program = load(main_arrangement_data["type"], is_meta=True)[0]
        meta_program_with_call = copy(main_meta_program)

        execute_result, modified_call = execute_with_context(meta_program_with_call, main_call or "", execute_results)

        global_logger.info("Phase 4: Processing result objects and assigning meshes...")
        combined_lookup = {**furniture_lookup, **arrangement_lookup}
        result_objs = extract_objects(execute_result["objs"], combined_lookup)
        global_logger.debug(f"Result objects: {result_objs}")
        arrangement_objs = []

        for obj_from_meta_program in result_objs:
            if isinstance(obj_from_meta_program, Obj):
                obj_to_add = obj_from_meta_program
                assign_mesh_to_object(obj_to_add, furniture_lookup, global_logger, normalize=True)
                arrangement_objs.append(obj_to_add)

        global_logger.info("Phase 5: Creating final Arrangement")
        final_arrangement = Arrangement(
            arrangement_objs,
            main_arrangement_data.get("description", "Generated arrangement"),
            modified_call or main_call or ""
        )

        # Attach hierarchy for later reuse
        try:
            setattr(final_arrangement, "_hierarchy", hierarchy)
        except Exception:
            pass

        global_logger.info(f"Successfully built arrangement with {len(arrangement_objs)} objects")
        # Return sub_arrangements in 2-tuple format
        sub_arrangements_2tuple = [(t, c) for t, c, _ in sub_arrangements]
        return True, final_arrangement, main_call, sub_arrangements_2tuple, execute_results

    except Exception as e:
        global_logger.error(f"Error building arrangement from JSON: {str(e)}")
        global_logger.error(f"Traceback: {traceback.format_exc()}")
        return False, None, None, None, None 