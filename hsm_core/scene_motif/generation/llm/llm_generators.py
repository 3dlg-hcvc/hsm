import json
import logging
import traceback
from copy import copy
from typing import List, Dict

import numpy as np

import hsm_core.vlm.gpt as gpt
from ...utils.library import load
from ...utils import calculate_arrangement_half_size
from ...utils.validation import inference_validation
from ...core.obj import Obj

logger = logging.getLogger(__name__)


async def generate_arrangement(inference_session, motif_type, meta_program, 
                               description, furniture_objs: List[Obj], 
                               is_leaf=False, sub_arrangements=None):
    """Generate a single arrangement or sub-arrangement."""
    try:
        if is_leaf:
            function_call_response = inference_session.send_with_validation("inference",
                {
                    "motif_type": motif_type,
                    "description": description,
                    "meta_program": meta_program.code_string,
                    "furniture_info": [f"{obj.label} (half_dimensions: {tuple(np.round(d, 3) for d in obj.bounding_box.half_size)})" for obj in furniture_objs]
                }, lambda response: inference_validation(response, meta_program))
        else:
            # Format sub-arrangements context with sizes
            arrangement_context = "Available sub-arrangements:\\n"
            if sub_arrangements:
                for idx, sub_arr in enumerate(sub_arrangements):
                    # Handle both 2-tuple (type, call) and 3-tuple (type, call, objects) formats
                    if len(sub_arr) == 3:
                        sub_type, sub_call, sub_objs_list = sub_arr
                    elif len(sub_arr) == 2:
                        sub_type, sub_call = sub_arr
                        sub_objs_list = []
                    else:
                        logger.warning(f"Unexpected sub_arrangement format: {sub_arr}")
                        continue

                    arrangement_size = calculate_arrangement_half_size(sub_objs_list)
                    arrangement_context += f"- {sub_type}: {sub_call}\\n"
                    arrangement_context += f"  Half size (w,h,d): {tuple(np.round(s, 3) for s in arrangement_size)}\\n"
                arrangement_context += "You MUST use 'sub_arrangements[i]' as label to reference pre-generated arrangements."
            else:
                arrangement_context = "No sub-arrangements available"
                
            # Use hierarchical inference for non-leaf nodes
            function_call_response = inference_session.send_with_validation("inference_hierarchical",
                {
                    "motif_type": motif_type,
                    "description": description,
                    "meta_program": meta_program.code_string,
                    "furniture_info": [f"{obj.label} (half_dimensions: {tuple(np.round(d, 3) for d in obj.bounding_box.half_size)})" for obj in furniture_objs],
                    "arrangement_context": arrangement_context
                }, lambda response: inference_validation(response, meta_program,
                                                                   variables={"sub_arrangements": sub_arrangements}),
                verbose=True)
        
        extracted_code = gpt.extract_code(function_call_response)
        logger.debug(f"Extracted code for {motif_type}: {extracted_code[:100]}...")
        return extracted_code
    except Exception as e:
        logger.error(f"Error generating arrangement: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ""  # Return empty string on error to prevent None


async def generate_arrangement_code(inference_session, arrangement_json, furniture_objs: List[Obj], *, furniture_lookup: Dict[str, Obj] | None = None):
    """Generate Python code strings for arrangements

    This function is responsible ONLY for interacting with the VLM to generate
    raw Python code strings. It does NOT handle execution, processing, or
    hierarchy building.

    Args:
        inference_session: VLM session
        arrangement_json: JSON with compositional arrangement specification
        furniture_objs: list of available furniture objects
        furniture_lookup: optional prebuilt lookup for furniture matching propagation

    Returns:
        Tuple of (main_call, sub_arrangements) where:
        - main_call: Python code string for the main arrangement
        - sub_arrangements: List of tuples (motif_type, code_string, []) for sub-arrangements
    """
    try:
        # PHASE 1: Parse and setup
        arrangement_json_data = json.loads(gpt.extract_json(arrangement_json))
        available_furniture = {obj.label: obj for obj in furniture_objs}

        # Check if we have validation feedback in past responses
        validation_feedback_found = any("VALIDATION FAILED" in response for response in inference_session.past_responses)
        if validation_feedback_found:
            logger.info("Found validation feedback in session history - VLM should be aware of previous errors")

        # PHASE 2: Generate code for nested arrangements (bottom-up)
        sub_arrangements = []
        if "elements" in arrangement_json_data:
            sub_arrangements = await _generate_nested_arrangement_codes(
                inference_session, arrangement_json_data["elements"], available_furniture, furniture_lookup=furniture_lookup
            )

        # PHASE 3: Generate main arrangement code
        main_call = await _generate_main_arrangement_code(
            inference_session, arrangement_json_data, available_furniture, sub_arrangements
        )

        logger.debug(f"Generated main_call: {main_call[:100] if main_call else 'None/Empty'}...")
        logger.debug(f"Generated {len(sub_arrangements)} sub_arrangements")

        return main_call, sub_arrangements

    except Exception as e:
        logger.error(f"Error generating arrangement code: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


async def _generate_nested_arrangement_codes(inference_session, elements, available_furniture, *, furniture_lookup: Dict[str, Obj] | None = None):
    """PHASE 2: Generate code for nested arrangements bottom-up"""
    all_arrangement_codes = await _generate_element_codes(
        inference_session, elements, available_furniture, furniture_lookup=furniture_lookup
    )
    # Sort by depth in descending order and extract type, code, and empty objects for final return value
    sorted_codes = sorted(all_arrangement_codes, key=lambda x: -x[2])  # depth is at index 2
    return [(t, c, []) for t, c, d in sorted_codes]  # Return (motif_type, code_string, [])


async def _generate_element_codes(inference_session, elements, available_furniture, depth=0, furniture_lookup: Dict[str, Obj] | None = None):
    """Generate Python code strings for elements"""
    local_arrangements = []

    # Build lookup only if not provided
    if furniture_lookup is None:
        furniture_lookup = {name: obj for name, obj in available_furniture.items()}
        for name, obj in list(available_furniture.items()):
            desc = (obj.description or obj.label).lower()
            furniture_lookup[desc] = obj
            if '_' in name:
                base_name = name.split('_')[0]
                furniture_lookup[base_name] = obj

    for element in elements:
        if element.get("type") == "object":
            continue

        # Generate code for nested elements
        sub_codes = []
        if "elements" in element:
            nested_codes = await _generate_element_codes(
                inference_session, element["elements"], available_furniture, depth + 1, furniture_lookup=furniture_lookup
            )
            # Extract codes and objects for this level
            sub_codes = [(t, c, []) for t, c, d in nested_codes if d == depth + 1]

        # Prepare furniture context for this element
        element_furniture_descriptions = set()
        for obj_desc_json in element.get("elements", []):
            if obj_desc_json.get("type") == "object":
                element_furniture_descriptions.add(obj_desc_json["description"].lower())

        # Find matching furniture objects (for context only, not execution)
        element_furniture_objs = []
        for desc_needed in element_furniture_descriptions:
            for name, obj_instance in available_furniture.items():
                obj_match_source = (obj_instance.description or obj_instance.label).lower()
                if desc_needed in obj_match_source:
                    element_furniture_objs.append(obj_instance)
                    break

        # Generate arrangement code
        loaded_programs = load(element["type"], is_meta=True)
        if not loaded_programs:
            logger.error(f"No meta program found for type: {element['type']}")
            raise ValueError(f"Missing meta program for arrangement type: {element['type']}")

        meta_program = loaded_programs[0]
        is_leaf = not any(e.get("type") != "object" for e in element.get("elements", []))

        arrangement_code = await generate_arrangement(
            inference_session,
            element["type"],
            meta_program,
            element.get("description", ""),
            element_furniture_objs,
            is_leaf=is_leaf,
            sub_arrangements=sub_codes
        )

        local_arrangements.append((element["type"], arrangement_code, depth))

    return local_arrangements


async def _generate_main_arrangement_code(inference_session, arrangement_json_data, available_furniture, sub_arrangements):
    """PHASE 3: Generate the main arrangement code"""
    main_furniture_objs_list = list(available_furniture.values())

    logger.info(f"Main arrangement furniture: {[f'{obj.label} (half_dimensions: {tuple(np.round(d, 3) for d in obj.bounding_box.half_size)})' for obj in main_furniture_objs_list]}")

    # Generate the main arrangement code
    main_meta_program = load(arrangement_json_data["type"], is_meta=True)[0]
    main_call = await generate_arrangement(
        inference_session,
        arrangement_json_data["type"],
        main_meta_program,
        arrangement_json_data["description"],
        main_furniture_objs_list,
        is_leaf=False,
        sub_arrangements=sub_arrangements
    )

    return main_call 