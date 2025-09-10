import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from hsm_core.utils import get_logger

import hsm_core.vlm.gpt as gpt
from ..core.arrangement import Arrangement

logger = get_logger('scene_motif.utils.saving')

if TYPE_CHECKING:
    from hsm_core.scene.specifications.object_spec import ObjectSpec


def _calculate_hierarchy_depth(arrangement_data: Dict, current_depth: int = 0) -> int:
    """Calculate the maximum depth of the hierarchical arrangement."""
    max_depth = current_depth
    
    if "elements" in arrangement_data:
        for element in arrangement_data["elements"]:
            if element.get("type") != "object" and "elements" in element:
                # This is a sub-arrangement, recurse
                element_depth = _calculate_hierarchy_depth(element, current_depth + 1)
                max_depth = max(max_depth, element_depth)
    
    return max_depth


def _count_total_objects(arrangement_data: Dict) -> int:
    """Count the total number of objects in the hierarchical arrangement."""
    total = 0
    
    if "elements" in arrangement_data:
        for element in arrangement_data["elements"]:
            if element.get("type") == "object":
                total += element.get("amount", 1)
            elif "elements" in element:
                # This is a sub-arrangement, recurse
                total += _count_total_objects(element)
    
    return total


def _extract_sub_arrangements(arrangement_data: Dict) -> List[Dict]:
    """Extract information about sub-arrangements in the hierarchy."""
    sub_arrangements = []
    
    if "elements" in arrangement_data:
        for element in arrangement_data["elements"]:
            if element.get("type") != "object" and "elements" in element:
                # This is a sub-arrangement
                sub_arrangements.append({
                    "type": element.get("type"),
                    "description": element.get("description"),
                    "object_count": _count_total_objects(element),
                    "depth": _calculate_hierarchy_depth(element)
                })
                # Recursively extract sub-arrangements from this element
                sub_arrangements.extend(_extract_sub_arrangements(element))
    
    return sub_arrangements


async def save_arrangement(
    final_arrangement: Arrangement,
    output_dir: Path,
    arrangement_id: str,
    furniture_specs: List["ObjectSpec"],
    save_prefix: str = "",
    arrangement_json: Optional[str] = None,
    main_call: Optional[str] = None,
    validate_response: Optional[Dict] = None,
    optimize: bool = False,
    make_tight: bool = False,
    save_pickle: bool = True,
    sub_arrangements: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[Path, Tuple[float, float, float], Arrangement]:
    """
    Save the arrangement and all related metadata to *program.json* together with the
    exported GLB mesh.

    Args:
        final_arrangement (Arrangement): The final arrangement object to be saved.
        output_dir (Path): The directory where the arrangement files will be saved.
        arrangement_id (str): Unique identifier for the arrangement.
        save_prefix (str): Optional prefix for the saved file names.
        furniture_specs (List[FurnitureSpec]): List of furniture specifications associated with the arrangement.
        description (str): Text description of the arrangement.
        arrangement_json (str, optional): JSON representation of the arrangement, if available.
        main_call (str, optional): The main function call associated with the arrangement.
        validate_response (Dict, optional): Validation response containing correctness and feedback.
        optimize: Whether the arrangement underwent spatial optimisation.
        make_tight: Whether objects were packed tightly during optimisation.
        save_pickle: Persist a legacy pickle file alongside the GLB export when `True` (default). Set to `False` to skip.
        sub_arrangements: Optional list of (motif_type, function_call) pairs for child motifs.

    Returns:
        Tuple consisting of:
            • Path to the saved GLB.
            • The arrangement extents `(width, length, height)`.
            • Arrangement instance.
    """
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    actual_extents = final_arrangement.get_extents(recalculate=True)
    # Convert numpy array to tuple, handle None case
    if actual_extents is not None:
        extents_tuple = (float(actual_extents[0]), float(actual_extents[1]), float(actual_extents[2]))
        extents_list = actual_extents.tolist()
    else:
        extents_tuple = (0.0, 0.0, 0.0)
        extents_list = list(extents_tuple)
    
    glb_path = save_dir / f"{arrangement_id}.glb"
    final_arrangement.save(str(glb_path))

    if save_pickle:
        pkl_path = save_dir / f"{arrangement_id}.pkl"
        final_arrangement.save_pickle(str(pkl_path))

    def _serialize_obj(obj):
        """Convert an Obj instance to a JSON-friendly dict (without heavy mesh data)."""
        try:
            bbox = obj.bounding_box
        except AttributeError:
            bbox = None

        return {
            "label": getattr(obj, "label", ""),
            "id": getattr(obj, "id", None),
            "description": getattr(obj, "description", ""),
            "mesh_path": getattr(obj, "mesh_path", None),
            "centroid": bbox.centroid.tolist() if bbox else None,
            "half_size": bbox.half_size.tolist() if bbox else None,
            "coord_axes": bbox.coord_axes.tolist() if bbox else None,
            "transform_matrix": getattr(obj, "transform_matrix", None).tolist() if hasattr(obj, "transform_matrix") else None,
        }

    arrangement_data = {
        "description": final_arrangement.description,
        "function_call": final_arrangement.function_call,
        "glb_path": str(glb_path),
        "extents": extents_list,
        "objects": [_serialize_obj(o) for o in final_arrangement.objs],
    }

    # Determine if this is a single object arrangement
    is_single_object = len(furniture_specs) == 1 and furniture_specs[0].amount == 1
    
    # Parse hierarchy information from arrangement_json
    hierarchy_info = None
    motif_type = "unknown"
    if arrangement_json:
        try:
            arrangement_json_data = json.loads(gpt.extract_json(arrangement_json))
            motif_type = arrangement_json_data.get("type", "unknown")
            
            # Extract hierarchical structure
            if "elements" in arrangement_json_data:
                hierarchy_info = {
                    "type": arrangement_json_data.get("type"),
                    "description": arrangement_json_data.get("description"),
                    "elements": arrangement_json_data["elements"],
                    "depth": _calculate_hierarchy_depth(arrangement_json_data),
                    "total_objects": _count_total_objects(arrangement_json_data),
                    "sub_arrangements": _extract_sub_arrangements(arrangement_json_data)
                }
        except Exception as e:
            logger.warning(f"Could not parse hierarchy from arrangement_json: {e}")
    
    # Ensure function_call value
    root_function_call = (
        "N/A" if is_single_object else (main_call or final_arrangement.function_call or "")
    )

    # Convert sub_arrangements list to serialisable form
    sub_arr_calls_serialised = []
    if sub_arrangements:
        for motif_t, call_str in sub_arrangements:
            sub_arr_calls_serialised.append({"motif_type": motif_t, "function_call": call_str})

    program_json = {
        "furniture_specs": [spec.to_dict() for spec in furniture_specs],
        "description": final_arrangement.description,
        "motif_type": "single_object" if is_single_object else motif_type,
        "function_call": root_function_call,
        "arrangement_json": arrangement_json,  # Raw VLM arrangement JSON (if available)
        "arrangement_data": arrangement_data,  # Rich serialised arrangement ready for reconstruction
        "extents": extents_list,
        "objects": [str(obj) for obj in final_arrangement.objs],
        "validation": validate_response or {"correct": 1, "feedback": "Single object arrangement - no spatial relationships to validate"},
        "optimize": optimize,
        "make_tight": make_tight,
        "hierarchy": hierarchy_info,
        "sub_arrangement_calls": sub_arr_calls_serialised,
    }
    
    with open(save_dir / "program.json", "w", encoding="utf-8") as f:
        json.dump(program_json, f, indent=4, ensure_ascii=False)

    return (glb_path, extents_tuple, final_arrangement) 