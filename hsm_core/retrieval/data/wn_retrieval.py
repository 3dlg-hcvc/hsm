"""
WordNet Retrieval Module

This module handles WordNet synset key retrieval and mapping for object descriptions.
"""

import json
from typing import List, Optional, Dict, Tuple
from hsm_core.utils import get_logger

import hsm_core.vlm.gpt as gpt
from hsm_core.config import PROMPT_DIR
from .data_utils import filter_hssd_categories, _load_hssd_alignment_data

logger = get_logger('retrieval.data.wn_retrieval')


def _get_wnsynsetkeys_for_labels(
    labels: List[str],
    filter_keys: List[str] | None = None,
    allow_none: bool = False
) -> List[str]:
    """
    Get the WordNet synset keys for the labels.

    Args:
        labels: List of object labels/descriptions
        filter_keys: Optional keys to filter the available synset keys
        allow_none: Whether to allow None values in the response

    Returns:
        List of WordNet synset keys corresponding to the labels
    """
    num_objs = len(labels)

    # Load available synset keys
    all_wnsynsetkeys_list: List[str] = list(_load_hssd_alignment_data().keys())

    # Apply filtering to limit the synset keys available to the VLM
    if filter_keys:
        all_wnsynsetkeys_list = [key for key in all_wnsynsetkeys_list if key in filter_keys]
        if not all_wnsynsetkeys_list:
            logger.warning(f"No synset keys found after filtering. Filter keys: {filter_keys[:5]}...")
            return [None] * num_objs

    def wnsynsetkeys_validation(response: str, allow_none_param: bool = False) -> Tuple[bool, str, int]:
        """Validate the WordNet synset keys response from VLM."""
        try:
            response_json: Dict[str, List[str]] = json.loads(gpt.extract_json(response))
        except json.JSONDecodeError as e:
            return False, f"Failed to decode the json response: {e}", -1

        valid = True
        error_message = ""

        retrieved_keys = response_json.get("wnsynsetkeys")
        if retrieved_keys is None or len(retrieved_keys) != num_objs:
            valid = False
            error_message = f"Expected {num_objs} wnsynsetkeys, got {len(retrieved_keys) if retrieved_keys is not None else 'None'}"
        else:
            for wnsynsetkey in retrieved_keys:
                if allow_none_param and str(wnsynsetkey).lower() == "none":
                    continue
                # Use the filtered list for validation
                if wnsynsetkey not in all_wnsynsetkeys_list:
                    valid = False
                    error_message = f"The WordNet synset key '{wnsynsetkey}' is invalid. Valid examples: {all_wnsynsetkeys_list[:10]}..."
                    break

        return valid, error_message, -1

    if not labels:
        return []

    logger.debug(f"Getting WN synset keys for labels: {labels}")
    if filter_keys:
        logger.debug(f"Filtering to {len(all_wnsynsetkeys_list)} synset keys for object type")

    wnsynsetkeys_session = gpt.Session(str(PROMPT_DIR / "retrieval_prompts.yaml"))
    def validation_wrapper(response: str) -> Tuple[bool, str, int]:
        return wnsynsetkeys_validation(response, allow_none)

    # Use the filtered synset keys list in the VLM prompt
    wnsynsetkeys_response = wnsynsetkeys_session.send_with_validation(
        "wnsynsetkeys",
        {"wnsynsetkeys": ",".join(all_wnsynsetkeys_list), "object_labels": ",".join(labels)},
        validation_wrapper, is_json=True
    )
    wnsynsetkeys_response_json: Dict[str, List[str]] = json.loads(gpt.extract_json(wnsynsetkeys_response))
    final_wnsynsetkeys = wnsynsetkeys_response_json.get("wnsynsetkeys", [])

    return final_wnsynsetkeys


def prepare_and_filter_candidates(
    objs_to_process: List,
    object_type
) -> Tuple[List[Optional[str]], List[List[str]]]:
    """
    Maps object descriptions to WN synset keys and filters mesh IDs by category.

    Args:
        objs_to_process: List of objects to process
        object_type: Type of object for filtering
        verbose: Enable debug output

    Returns:
        Tuple of (wnsynsetkeys, filtered_embedding_indices_for_obj)
    """
    obj_descriptions_list = [obj.description or obj.label for obj in objs_to_process]

    # WN Synset Key Retrieval
    try:
        wnsynsetkeys = _get_wnsynsetkeys_for_labels(
            obj_descriptions_list,
            filter_keys=filter_hssd_categories(object_type),
            allow_none=False
        )
        logger.info(f"WordNet synset keys: {wnsynsetkeys}")
    except Exception as e:
        logger.error(f"Error getting WN Synset Keys: {e}. Proceeding without strict filtering.")
        wnsynsetkeys = [None] * len(objs_to_process)

    # Create filtered embedding index based on WordNet synsets
    filtered_embedding_indices_for_obj = []
    hssd_data = _load_hssd_alignment_data()

    for obj_idx, obj_iter in enumerate(objs_to_process):
        valid_mesh_ids = []
        if (obj_idx < len(wnsynsetkeys) and
            wnsynsetkeys[obj_idx] is not None and
            wnsynsetkeys[obj_idx] in hssd_data):
            valid_mesh_ids = [row["id"] for row in hssd_data[wnsynsetkeys[obj_idx]]]
        else:
            logger.warning(f"No valid WN synset key found or index missing for '{obj_iter.label}'. Cannot filter by WN key.")

        if not valid_mesh_ids:
            logger.warning(f"No valid meshes found via WN key for {obj_iter.label}")

        filtered_embedding_indices_for_obj.append(valid_mesh_ids)

    return wnsynsetkeys, filtered_embedding_indices_for_obj
