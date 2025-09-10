"""
This module provides shared functions for assigning meshes to objects
"""

from copy import copy
import logging
from typing import Dict, List
from hsm_core.utils import get_logger

from ..core.obj import Obj

logger = get_logger('scene_motif.utils.mesh_utils')


def create_furniture_lookup(furniture_specs, retrieved_furniture: List[Obj]) -> Dict[str, Obj]:
    """Create comprehensive furniture lookup dictionary."""
    furniture_lookup = {}

    # Create mapping of spec names to their normalized forms
    spec_names = {spec.name.lower(): spec.name.lower() for spec in furniture_specs}

    # Create a more robust lookup that handles variations in naming
    for obj in retrieved_furniture:
        # Store by original object label
        furniture_lookup[obj.label.lower()] = obj

        # Also store by base name without suffixes (for multiple instances)
        base_name = obj.label.lower()
        if '_' in base_name:
            base_name = base_name.split('_')[0]
            furniture_lookup[base_name] = obj

        # Match with furniture specs (to link spec names to retrieved Objs if needed elsewhere, though not directly for dimensions here)
        for spec_name_lower in spec_names:
            if spec_name_lower in obj.label.lower() or obj.label.lower() in spec_name_lower:
                furniture_lookup[spec_name_lower] = obj

    logger.debug("Furniture lookup mapping for mesh assignment:")
    for key, obj in furniture_lookup.items():
        logger.debug(f"- {key} -> {obj.label} (mesh: {getattr(obj, 'mesh_path', 'None')})")

    return furniture_lookup


def assign_mesh_to_object(obj: Obj, lookup: Dict[str, Obj], logger: logging.Logger, normalize: bool = False) -> bool:
    """Assign mesh to object from lookup table using configurable matching strategies."""
    if normalize:
        return _assign_mesh_with_normalization(obj, lookup, logger)
    else:
        return _assign_mesh_simple(obj, lookup, logger)


def _assign_mesh_simple(obj: Obj, lookup: Dict[str, Obj], logger: logging.Logger) -> bool:
    """Simple mesh assignment using basic matching strategies."""
    obj_key = obj.label.lower()

    # Strategy 1: Exact match
    if obj_key in lookup:
        matching_furniture = lookup[obj_key]
        logger.debug(f"Found mesh for {obj.label} -> {matching_furniture.mesh_path}")
        obj.mesh = copy(matching_furniture.mesh)
        obj.mesh_path = matching_furniture.mesh_path
        return True

    # Strategy 2: Partial matches (contains/in)
    for key, value in lookup.items():
        if key in obj_key or obj_key in key:
            logger.debug(f"Found partial match for {obj.label} -> {key} -> {value.mesh_path}")
            obj.mesh = copy(value.mesh)
            obj.mesh_path = value.mesh_path
            return True

    logger.debug(f"No mesh found for {obj.label}")
    logger.debug(f"Available keys: {list(lookup.keys())}")
    return False


def _validate_and_fix_mesh(obj: Obj, found_matching_retrieved_obj, logger: logging.Logger) -> None:
    """Validate and fix mesh after assignment"""
    if isinstance(obj.mesh, list):
        if obj.mesh and isinstance(obj.mesh[0], __import__('trimesh').Trimesh):
            obj.mesh = obj.mesh[0]
        elif not obj.mesh:
            logger.warning(f"Copy of mesh for {obj.label} resulted in an empty list. Setting mesh to None.")
            obj.mesh = None
        else:
            logger.warning(f"Copy of mesh for {obj.label} resulted in an unexpected list content: {type(obj.mesh[0] if obj.mesh else 'empty list')}. Setting mesh to None.")
            obj.mesh = None
    elif obj.mesh is None and found_matching_retrieved_obj.mesh is not None:
        logger.warning(f"Copy of a non-None mesh for {obj.label} resulted in None. This is unexpected. Mesh path: {obj.mesh_path}")


def _assign_mesh_with_normalization(obj: Obj, lookup: Dict[str, Obj], logger: logging.Logger) -> bool:
    """Assign mesh using normalized matching (spaces/underscores)."""
    keys_to_search = set()
    current_obj_label_lower = obj.label.lower()
    keys_to_search.add(current_obj_label_lower)
    keys_to_search.add(current_obj_label_lower.replace(" ", "").replace("_", ""))

    # Attempt to find a base name if there's a suffix like _1, _2
    if '_' in current_obj_label_lower:
        name_parts = current_obj_label_lower.rsplit('_', 1)
        if len(name_parts) == 2 and name_parts[1].isdigit():
            base_label_candidate = name_parts[0]
            keys_to_search.add(base_label_candidate)
            keys_to_search.add(base_label_candidate.replace("_", " "))

    found_matching_retrieved_obj = None
    found_key_for_mesh = None

    # Strategy 1: Direct match with current_obj_label_lower
    if current_obj_label_lower in lookup:
        candidate_obj = lookup[current_obj_label_lower]
        if hasattr(candidate_obj, 'mesh') and candidate_obj.mesh is not None:
            found_matching_retrieved_obj = candidate_obj
            found_key_for_mesh = current_obj_label_lower

    # Strategy 2: Iterate through all generated keys if direct match failed or had no mesh
    if not found_matching_retrieved_obj:
        # Sort keys to try more specific (longer) ones first, then fall back to shorter (base) names.
        for key_try in sorted(list(keys_to_search), key=len, reverse=True):
            if key_try == current_obj_label_lower and found_key_for_mesh == current_obj_label_lower:
                continue  # Already tried this key if direct match was attempted
            if key_try in lookup:
                candidate_obj = lookup[key_try]
                if hasattr(candidate_obj, 'mesh') and candidate_obj.mesh is not None:
                    found_matching_retrieved_obj = candidate_obj
                    found_key_for_mesh = key_try
                    logger.debug(f"Mesh found for '{current_obj_label_lower}' via alternative key '{found_key_for_mesh}'")
                    break  # Found a suitable mesh with this alternative key

    if found_matching_retrieved_obj:
        logger.debug(f"Assigning mesh for '{obj.label}' from furniture_lookup key '{found_key_for_mesh}' (source: {found_matching_retrieved_obj.label}, mesh: {getattr(found_matching_retrieved_obj, 'mesh_path', 'N/A')})")
        obj.mesh = copy(found_matching_retrieved_obj.mesh)
        obj.mesh_path = found_matching_retrieved_obj.mesh_path

        _validate_and_fix_mesh(obj, found_matching_retrieved_obj, logger)

        return True
    else:
        current_obj_had_mesh = hasattr(obj, 'mesh') and obj.mesh is not None
        if not current_obj_had_mesh:
            logger.warning(f"No mesh assigned for '{obj.label}'. No suitable mesh found in furniture_lookup. Searched keys derived from label: {sorted(list(keys_to_search))}. Furniture_lookup example keys: {list(lookup.keys())[:10]}")

        return False


# Backward compatibility aliases
def assign_mesh_to_object_with_normalization(obj: Obj, lookup: Dict[str, Obj], logger: logging.Logger) -> bool:
    """Alias for assign_mesh_to_object with normalize=True."""
    return assign_mesh_to_object(obj, lookup, logger, normalize=True)
