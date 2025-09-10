import json
import yaml
import traceback
from typing import Tuple, Dict, Optional, List
from copy import deepcopy
from hsm_core.utils import get_logger

import hsm_core.vlm.gpt as gpt
from hsm_core.scene_motif.programs import validator

from ..core.arrangement import Arrangement
from ..programs.program import Program

from hsm_core.config import PROMPT_DIR
MOTIF_DEFINITIONS_PATH = PROMPT_DIR / "motif_definitions.yaml"

logger = get_logger('scene_motif.utils.validation')

try:
    with open(MOTIF_DEFINITIONS_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    MOTIF_CONSTRAINTS_DATA = data['constraints']
    ALL_MOTIFS_FROM_DATA = set(MOTIF_CONSTRAINTS_DATA.keys()) if MOTIF_CONSTRAINTS_DATA else set()
    if not ALL_MOTIFS_FROM_DATA:
        logger.warning("ALL_MOTIFS_FROM_DATA is empty. Validation might not work as expected.")
except (FileNotFoundError, ValueError) as e:
    logger.error(f"Failed to load motif constraints: {e}")
    raise RuntimeError(f"Could not load motif constraints from {MOTIF_DEFINITIONS_PATH}. {e}") from e


def validate_remaining_arrangements(response: str, remaining_objects: list) -> Tuple[bool, str, int]:
    """
    Validate that secondary arrangements use all remaining furniture items.
    
    Args:
        response (str): The JSON response containing arrangement information
        remaining_objects (list): List of remaining furniture objects to be assigned
        
    Returns:
        Tuple[bool, str, int]: (is_valid, error_message, error_code)
    """
    try:
        # Parse the response JSON
        arrangement_data = json.loads(gpt.extract_json(response))
        if isinstance(arrangement_data, dict):
            arrangement_data = [arrangement_data]

        # If no arrangements but objects remain, error out early
        if not arrangement_data and remaining_objects:
            return False, "No arrangements provided but objects remain", 1

        # Collect all used object names (lowercased)
        used_objects = set()
        for arrangement in arrangement_data:
            objs = arrangement.get("objects") or {}
            used_objects.update(name.lower() for name in objs.keys())

        # Determine which remain unassigned
        final_remaining = [obj for obj in remaining_objects if obj.lower() not in used_objects]

        # Validate remaining furniture
        if not final_remaining:
            return True, "", -1
        else:
            return False, f"Unassigned furniture items remain: {', '.join(final_remaining)}", 1

    except json.JSONDecodeError:
        return False, "Invalid JSON format in response", 3
    except Exception as e:
        return False, f"Unexpected error during validation: {e}", 4

def is_sm_exceeds_support_region(
    arrangement: Arrangement, 
    support_surface_constraints: Dict[str, Dict]
) -> bool:
    """
    Check if the arrangement's extents exceed the available support surface constraints.
    
    Args:
        arrangement: The generated arrangement to check
        support_surface_constraints: Dictionary mapping object labels to their support surface constraints
        
    Returns:
        bool: True if arrangement exceeds any support surface constraints, False otherwise
    """
    if not arrangement or not arrangement.objs or not support_surface_constraints:
        return False
    
    arrangement_extents = arrangement.get_extents(recalculate=True)
    if arrangement_extents is None:
        return False
    
    # Calculate arrangement footprint area (X * Z dimensions)
    arrangement_footprint_area = arrangement_extents[0] * arrangement_extents[2]
    
    # Check against each object's support surface constraints
    for obj in arrangement.objs:
        obj_label = obj.label.lower() if hasattr(obj, 'label') else str(obj).lower()
        
        # Find matching constraint (try exact match first, then partial match)
        constraint_data = None
        if obj_label in support_surface_constraints:
            constraint_data = support_surface_constraints[obj_label]
        else:
            # Try partial matching for object labels
            for constraint_label, constraint_info in support_surface_constraints.items():
                if constraint_label in obj_label or obj_label in constraint_label:
                    constraint_data = constraint_info
                    break
        
        if constraint_data:
            # Check available surface area constraint
            available_area = constraint_data.get('available_area', float('inf'))
            if available_area != float('inf') and arrangement_footprint_area > available_area:
                logger.info(f"Arrangement footprint area ({arrangement_footprint_area:.3f}m²) exceeds available surface area ({available_area:.3f}m²) for {obj_label}")
                return True
            
            # Check surface bounds constraints
            surface_bounds = constraint_data.get('bounds')
            if surface_bounds:
                surface_width = surface_bounds.get('width', float('inf'))
                surface_depth = surface_bounds.get('depth', float('inf'))
                
                if (arrangement_extents[0] > surface_width or arrangement_extents[2] > surface_depth):
                    logger.info(f"Arrangement extents ({arrangement_extents[0]:.3f}m × {arrangement_extents[2]:.3f}m) exceed surface bounds ({surface_width:.3f}m × {surface_depth:.3f}m) for {obj_label}")
                    return True
            
            # Check height clearance constraint
            max_height = constraint_data.get('max_height', float('inf'))
            if max_height != float('inf') and arrangement_extents[1] > max_height:
                logger.info(f"Arrangement height ({arrangement_extents[1]:.3f}m) exceeds max height ({max_height:.3f}m) for {obj_label}")
                return True
    
    return False 

def validate_compositional_json(response: str) -> tuple[bool, str, int]:
    """
    Validate the arrangement JSON response format and content, including motif-specific
    element counts with constraints.

    Args:
        response (str): The JSON response containing arrangement information

    Returns:
        Tuple[bool, str, int]: (is_valid, error_message, error_code)
    """

    try:
        layout_data = json.loads(gpt.extract_json(response))

        # -- Basic shape checks --
        for key in ("type", "description", "elements"):
            if key not in layout_data:
                return False, f"Missing required key: {key}", 1

        if not isinstance(layout_data["type"], str) or not layout_data["type"].strip():
            return False, "Type must be a non-empty string", 2

        if not isinstance(layout_data["description"], str) or not layout_data["description"].strip():
            return False, "Description must be a non-empty string", 3

        if not isinstance(layout_data["elements"], list):
            return False, "Elements must be a list", 4

        motif_type = layout_data["type"]
        
        # -- Check that motif_type is recognized and get its constraints --
        if motif_type not in MOTIF_CONSTRAINTS_DATA:
            return False, f"Unknown or unsupported motif type '{motif_type}'", 8
        
        constraints = MOTIF_CONSTRAINTS_DATA[motif_type]
        min_elements = constraints.get('min_unique_types', 1) # Default to 1 if not specified
        max_elements = constraints.get('max_unique_types', float('inf')) # Default to no upper limit if not specified

        elements = layout_data["elements"]
        num_elements = len(elements)

        # -- Verify the number of element types matches the motif's requirement --
        if not (min_elements <= num_elements <= max_elements):
            if min_elements == max_elements:
                expected_str = f"{min_elements}"
            else:
                expected_str = f"between {min_elements} and {max_elements}"
            return (
                False,
                f"Motif '{motif_type}' requires {expected_str} unique element(s)/group(s), "
                f"found {num_elements}",
                9
            )

        # -- Validate each element, recursing on nested motifs --
        for idx, element in enumerate(elements, start=1):
            if not isinstance(element, dict):
                return False, f"Element #{idx} must be a dictionary", 5

            elem_type = element.get("type")
            if not isinstance(elem_type, str) or not elem_type.strip(): # check strip
                return False, f"Element #{idx} missing a valid 'type' (must be non-empty string)", 10

            # Case A: simple object
            if elem_type.lower() == "object":
                amount = element.get("amount")
                if not isinstance(amount, int) or amount < 1:
                    return False, (
                        f"Element #{idx} ('{element.get('description', 'object')}') must have a positive integer 'amount'"
                    ), 11
                desc = element.get("description")
                if not isinstance(desc, str) or not desc.strip():
                    return False, (
                        f"Element #{idx} ('object') must have a non-empty 'description'"
                    ), 12

            # Case B: nested motif
            elif elem_type in ALL_MOTIFS_FROM_DATA:
                nested_json_str = json.dumps(element) # This is the string to pass for recursion
                valid, msg, code = validate_compositional_json(nested_json_str) # Pass string, not dict
                if not valid:
                    return False, f"Nested motif in element #{idx} ('{elem_type}') invalid: {msg}", 13

            else:
                return False, f"Unknown element type '{elem_type}' in element #{idx}", 10

        return True, "", -1

    except json.JSONDecodeError:
        return False, "Invalid JSON format in response", 6
    except Exception as e:
        import traceback
        logger.error(f"Unexpected error during validation: {e}\n{traceback.format_exc()}")
        return False, f"Unexpected error during validation: {e}", 7

def inference_validation(response: str, meta_program: Program, variables: Optional[dict] = None, expected_objects: Optional[List[str]] = None) -> tuple[bool, str, int]:
    """
    Validate the inference response by checking syntax and object availability.

    Args:
        response (str): The response string from the VLM
        meta_program (Program): The meta-program being used
        variables (dict): Available variables for validation
        expected_objects (list, optional): List of available object labels that the VLM is allowed to use

    Returns:
        tuple[bool, str, int]:
            - bool: True if response is valid, False otherwise
            - str: Error message if validation fails, empty string if successful
            - int: -1 if successful, 0 if validation fails
    """
    try:
        # Extract the function call from the response
        function_call = gpt.extract_code(response)
        if not function_call:
            return False, "Error: No valid function call found in response", 0

        # Create a test program by combining the meta-program and function call
        test_program = deepcopy(meta_program)
        test_program.append_code(f"objs = {function_call}")

        # Validate syntax
        valid, error_message = validator.validate_syntax(test_program, variables=variables)
        if not valid:
            return False, f"Syntax error: {error_message}", 0

        # If expected_objects is provided, validate that only available objects are used
        if expected_objects is not None:
            try:
                # Execute the program to get the created objects
                variable_dict = validator.execute(test_program)
                if "objs" not in variable_dict:
                    return False, "Program does not produce 'objs' variable", 0

                created_objs = variable_dict["objs"]

                # Extract labels from created objects
                created_labels = []
                for obj in created_objs:
                    if hasattr(obj, 'label'):
                        created_labels.append(obj.label.lower())
                    elif isinstance(obj, dict) and 'label' in obj:
                        created_labels.append(obj['label'].lower())

                # Extract available labels (normalize to lowercase)
                available_labels = [label.lower() for label in expected_objects]

                # Check for unavailable objects that are referenced in the generated code
                unavailable_objects = []
                for created_label in created_labels:
                    # Skip sub_arrangement references as they're handled separately
                    if created_label.startswith('sub_arrangements['):
                        continue

                    found = False
                    for available_label in available_labels:
                        # Exact match only
                        if available_label == created_label:
                            found = True
                            break

                    if not found:
                        unavailable_objects.append(created_label)

                if unavailable_objects:
                    return False, f"Program references unavailable objects: {', '.join(unavailable_objects)}. Available objects: {', '.join(available_labels)}", 0

            except Exception as obj_validation_error:
                # Log the error but don't fail validation - mesh issues shouldn't break optimization
                logger.warning(f"Object availability validation failed (continuing anyway): {obj_validation_error}")
                # Continue without failing - this prevents mesh-related optimization breaks

        return True, "", -1

    except Exception as e:
        return False, f"Validation error: {str(e)}\nTraceback: {traceback.format_exc()}", 0