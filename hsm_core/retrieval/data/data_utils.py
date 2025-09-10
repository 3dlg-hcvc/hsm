"""
Data Utilities for Retrieval Module

This module provides utilities for loading and filtering HSSD data,
including WordNet synset key filtering and category-based filtering.
"""

import json
from typing import List, Optional, Dict, Tuple, Set
from hsm_core.utils import get_logger

from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.config import DATA_PATH

logger = get_logger('retrieval.data.utils')


_HSSD_ALIGNMENT_DATA: Optional[Dict] = None
_OBJECT_CATEGORIES_DATA: Optional[Dict] = None


def _load_hssd_alignment_data() -> Dict:
    """Load HSSD alignment data if not already loaded."""
    global _HSSD_ALIGNMENT_DATA
    if _HSSD_ALIGNMENT_DATA is None:
        hssd_index_path = DATA_PATH / "preprocessed" / "hssd_wnsynsetkey_index.json"
        try:
            with open(hssd_index_path, 'r') as f:
                _HSSD_ALIGNMENT_DATA = json.load(f)
            logger.info(f"Successfully loaded HSSD alignment data from {hssd_index_path}")
        except Exception as e:
            raise Exception(f"Error loading HSSD alignment data from {hssd_index_path}: {e}")
    return _HSSD_ALIGNMENT_DATA


def _load_object_categories_data() -> Dict:
    """Load object categories data if not already loaded."""
    global _OBJECT_CATEGORIES_DATA
    if _OBJECT_CATEGORIES_DATA is None:
        categories_path = DATA_PATH / "preprocessed" / "object_categories.json"
        try:
            with open(categories_path, 'r') as f:
                _OBJECT_CATEGORIES_DATA = json.load(f)
            logger.info(f"Successfully loaded object categories data from {categories_path}")
        except Exception as e:
            raise Exception(f"Error loading object categories data from {categories_path}: {e}")
    return _OBJECT_CATEGORIES_DATA


def filter_hssd_categories(object_type_input: ObjectType | str) -> List[str]:
    """Filter the HSSD categories for the given object type."""
    # Handle ObjectType enum directly
    if isinstance(object_type_input, ObjectType):
        category_string_key = OBJECT_TYPE_MAPPING.get(object_type_input, object_type_input.value)
    else:
        # Handle string input
        if object_type_input.startswith("ObjectType."):
            member_name = object_type_input.split(".", 1)[1]
            try:
                obj_type = ObjectType[member_name]
                category_string_key = OBJECT_TYPE_MAPPING.get(obj_type, obj_type.value)
            except KeyError:
                logger.warning(f"Invalid ObjectType member '{member_name}'. Defaulting to UNDEFINED.")
                category_string_key = OBJECT_TYPE_MAPPING[ObjectType.UNDEFINED]
        else:
            # Try to find matching enum value
            for member in ObjectType:
                if member.value == object_type_input:
                    category_string_key = OBJECT_TYPE_MAPPING.get(member, member.value)
                    break
            else:
                logger.warning(f"String value '{object_type_input}' does not match any ObjectType. Defaulting to UNDEFINED.")
                category_string_key = OBJECT_TYPE_MAPPING[ObjectType.UNDEFINED]

    categories_data = _load_object_categories_data()
    return categories_data.get(category_string_key, [])


def get_fallback_mesh_ids(
    label: str,
    object_type: ObjectType = ObjectType.UNDEFINED
) -> List[Tuple[str, Set[str]]]:
    """
    Get fallback mesh IDs for an object label.

    Args:
        label: Object label to find fallback meshes for
        object_type: Type of object for broader fallback

    Returns:
        List of tuples (search_type, mesh_ids_set) ordered by preference
    """
    hssd_data = _load_hssd_alignment_data()
    type_specific_wn_keys = filter_hssd_categories(object_type)

    # Collect all mesh IDs for this object type
    mesh_ids = set()
    for wn_key in type_specific_wn_keys:
        if wn_key in hssd_data:
            for row_data in hssd_data[wn_key]:
                mesh_ids.add(row_data["id"])

    return [("object_type", mesh_ids)] if mesh_ids else []


# Object type mapping (keeping for backward compatibility)
OBJECT_TYPE_MAPPING = {
    ObjectType.LARGE: "large_objects",
    ObjectType.WALL: "wall_objects",
    ObjectType.SMALL: "small_objects",
    ObjectType.CEILING: "ceiling_objects",
    ObjectType.UNDEFINED: "undefined"
}



