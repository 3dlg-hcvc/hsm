import time
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
from copy import deepcopy
import numpy as np
from pathlib import Path
from hsm_core.utils import get_logger

from ..core.obj import Obj
from ..core.arrangement import Arrangement
from .mesh_utils import assign_mesh_to_object_with_normalization

if TYPE_CHECKING:
    from hsm_core.scene.core.motif import SceneMotif
    from hsm_core.scene.specifications.object_spec import ObjectSpec

logger = get_logger('scene_motif.utils.utils')


def log_time(start_time: float, message: str, indent_level: int = 0):
    """Log time taken for an operation with indentation."""
    elapsed = time.time() - start_time
    indent = "  " * indent_level
    logger.info(f"{indent}{message}: {elapsed:.2f}s")

def calculate_arrangement_half_size(objects):
    """Calculate the half size of an arrangement based on object dimensions."""
    if not objects:
        return (0, 0, 0)
    
    # Check if any objects have valid transform matrices
    valid_objects = []
    for obj in objects:
        if isinstance(obj, Obj) and obj.transform_matrix is not None:
            valid_objects.append(obj)
    
    if not valid_objects:
        logger.warning("No objects have transform matrices, using bounding box fallback")
        max_half_size = [0.0, 0.0, 0.0]
        for obj in objects:
            if isinstance(obj, Obj):
                half_size = obj.bounding_box.half_size
                for i in range(3):
                    max_half_size[i] = max(max_half_size[i], half_size[i])
        return tuple(max_half_size)
    
    min_coords = np.array([float('inf')] * 3)
    max_coords = np.array([float('-inf')] * 3)
    
    for obj in valid_objects:
        half_size = obj.bounding_box.half_size
        transform = obj.transform_matrix
        
        corners = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ]) * half_size.reshape(1, 3)
        
        # Transform corners to world space
        corners_homogeneous = np.hstack([corners, np.ones((8, 1))])
        transformed_corners = (transform @ corners_homogeneous.T).T[:, :3]
        
        min_coords = np.minimum(min_coords, transformed_corners.min(axis=0))
        max_coords = np.maximum(max_coords, transformed_corners.max(axis=0))
    
    full_size = max_coords - min_coords
    half_size = full_size / 2
    return tuple(half_size)

def extract_objects(result, lookup: Optional[dict] = None) -> List[Obj]:
    """Extract objects from nested arrangements"""
    logger.debug(f"Extracting objects from {result.__class__.__name__}")
    
    if isinstance(result, Arrangement):
        logger.debug(f"Processing arrangement containing {len(result.objs)} objects")
        all_objs = []
        
        for obj in result.objs:
            obj_name = getattr(obj, 'label', str(obj))
            logger.debug(f"Processing {obj_name}")
            
            if obj_name.startswith('sub_arrangements['):
                try:
                    # Try lookup table first
                    if lookup is not None and obj_name in lookup:
                        logger.debug(f"Found arrangement in lookup for {obj_name}")
                        nested_objs = extract_objects(lookup[obj_name], lookup)
                        logger.debug(f"Extracted objects from lookup: {[o.label for o in nested_objs]}")
                        all_objs.extend(nested_objs)
                    # Fall back to referenced arrangement
                    elif hasattr(obj, '_referenced_arrangement'):
                        logger.debug(f"Found referenced arrangement")
                        nested_objs = extract_objects(obj._referenced_arrangement, lookup)
                        logger.debug(f"Extracted objects from reference: {[o.label for o in nested_objs]}")
                        all_objs.extend(nested_objs)
                    else:
                        logger.warning(f"No arrangement found for {obj_name}")
                except Exception as e:
                    logger.error(f"Error processing sub-arrangement: {e}")
            else:
                new_obj = deepcopy(obj)
                if lookup is not None:
                    assign_mesh_to_object_with_normalization(new_obj, lookup, logger)
                all_objs.append(new_obj)
        return all_objs
        
    elif isinstance(result, dict):
        logger.debug("Processing dictionary of arrangements")
        all_objs = []
        for key, arr in result.items():
            logger.debug(f"Processing key: {key}")
            all_objs.extend(extract_objects(arr, lookup))
        return all_objs
        
    elif isinstance(result, (list, tuple)):
        logger.debug(f"Processing {type(result).__name__} of {len(result)} items")
        all_objs = []
        for item in result:
            if isinstance(item, (Arrangement, dict, list, tuple)) or hasattr(item, 'objs'):
                logger.debug(f"Processing nested {item.__class__.__name__}")
                all_objs.extend(extract_objects(item, lookup))
            elif isinstance(item, Obj):
                logger.debug(f"Found object: {item.label}")
                all_objs.append(item)
        return all_objs
        
    logger.debug(f"Processing single item of type {result.__class__.__name__}")
    return [result] if isinstance(result, Obj) else []

def resolve_sub_arrangements(call_string, execute_results):
    """Recursively resolve all sub-arrangement references in a call string."""
    modified = call_string
    for i in range(len(execute_results)):
        # Handle both string and direct object references
        if f"'sub_arrangements[{i}]'" in modified:
            modified = modified.replace(
                f"'sub_arrangements[{i}]'",
                f"execute_results[{i}]"
            )
        elif f"sub_arrangements[{i}]" in modified:
            modified = modified.replace(
                f"sub_arrangements[{i}]",
                f"execute_results[{i}]"
            )
    return modified 

async def persist_motif_arrangement(
    motif: "SceneMotif",
    *,
    final_arrangement: Arrangement,
    output_dir: Path,
    arrangement_id: str,
    furniture_specs: List["ObjectSpec"],
    arrangement_json: Optional[str] = None,
    validate_response: Optional[Dict] = None,
    main_call: Optional[str] = None,
    sub_arrangements: Optional[List[Tuple[str, str]]] = None,
    save_pickle: bool = True,
    optimize: bool = True,
    make_tight: bool = False,
) -> "SceneMotif":
    """Persist arrangement to disk and update motif with resulting metadata."""

    from .saving import save_arrangement

    glb_path, extents, arrangement_result = await save_arrangement(
        final_arrangement=final_arrangement,
        output_dir=output_dir,
        arrangement_id=arrangement_id,
        furniture_specs=furniture_specs,
        arrangement_json=arrangement_json,
        main_call=main_call,
        sub_arrangements=sub_arrangements,
        validate_response=validate_response,
        save_pickle=save_pickle,
        optimize=optimize,
        make_tight=make_tight,
    )

    motif.glb_path = str(glb_path)
    motif.extents = extents
    motif.arrangement = arrangement_result

    # These attributes are only set in motif_processing; keep conditional
    if arrangement_json is not None:
        try:
            setattr(motif, "arrangement_json", arrangement_json)
        except AttributeError:
            pass

    if validate_response is not None:
        try:
            setattr(motif, "validate_response", validate_response)
        except AttributeError:
            pass

    return motif 