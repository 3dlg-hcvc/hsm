import json
from typing import Tuple, List, Dict, Any, Optional, Set
from shapely.geometry import Polygon, Point
from collections import Counter

from hsm_core.vlm.gpt import extract_json
from hsm_core.scene.core.spec import ObjectSpec
from hsm_core.scene.core.motif import SceneMotif


def _parse_json_response(response: str) -> Tuple[Dict[str, Any], Tuple[bool, str, int]]:
    """Parse JSON response with consistent error handling.

    Args:
        response: JSON string response

    Returns:
        Tuple of (parsed_data, error_info). If parsing fails, parsed_data is None.
    """
    try:
        return json.loads(extract_json(response)), (True, "", -1)
    except json.JSONDecodeError:
        return None, (False, "Invalid JSON format in response", 4)
    except Exception as e:
        return None, (False, f"Unexpected error parsing JSON: {str(e)}", 8)


def _validate_required_fields(data: Dict[str, Any], required_fields: Set[str], context: str = "") -> List[str]:
    """Validate that required fields are present in data.

    Args:
        data: Dictionary to validate
        required_fields: Set of required field names
        context: Context string for error messages

    Returns:
        List of error messages
    """
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        prefix = f"{context}: " if context else ""
        return [f"{prefix}Missing required fields: {', '.join(missing_fields)}"]
    return []


def _validate_list_format(data: Any, expected_length: int, field_name: str) -> List[str]:
    """Validate that data is a list of expected length.

    Args:
        data: Data to validate
        expected_length: Expected length of list
        field_name: Field name for error messages

    Returns:
        List of error messages
    """
    if not isinstance(data, list):
        return [f"{field_name} must be a list"]
    if len(data) != expected_length:
        return [f"{field_name} must be a list of {expected_length} elements"]
    return []


def _validate_numeric_list(data: List[Any], field_name: str, allow_zero: bool = False) -> List[str]:
    """Validate that list contains only positive numbers.

    Args:
        data: List to validate
        field_name: Field name for error messages
        allow_zero: Whether to allow zero values

    Returns:
        List of error messages
    """
    if not all(isinstance(coord, (int, float)) for coord in data):
        return [f"{field_name} coordinates must be numbers"]

    if not allow_zero and not all(coord > 0 for coord in data):
        return [f"{field_name} values must be positive numbers"]

    return []

def validate_furniture_layout(response: str, room_polygon: Polygon, scene_motifs: List[SceneMotif]) -> Tuple[bool, str, int]:
    """
    Validate the furniture layout response from the VLM.

    Args:
        response: The JSON string response from the VLM.
        room_polygon: The Shapely Polygon representing the room's shape.
        scene_motifs: List of scene motifs to validate against.

    Returns:
        Tuple of (is_valid, error_message, error_code).
        Error codes: -1=success, 1=missing positions key, 2=missing motif ids,
        3=validation errors, 4=json error, 5=key error, 6=unexpected error.
    """
    layout_data, parse_result = _parse_json_response(response)
    if not parse_result[0]:
        return parse_result

    if "positions" not in layout_data:
        return False, "Missing 'positions' key in the response.", 1

    furniture_layout = layout_data["positions"]
    layout_ids = set(item["id"] for item in furniture_layout)
    expected_ids = set(motif.id for motif in scene_motifs)
    missing_ids = expected_ids - layout_ids

    if missing_ids:
        return False, f"Missing id in the layout: {', '.join(missing_ids)}", 2

    errors = []
    for item in furniture_layout:
        motif_id = item["id"]
        position = item.get("position")
        rotation = item.get("rotation")
        rationale = item.get("rationale")
        ignore_collision = item.get("ignore_collision")
        wall_alignment = item.get("wall_alignment", False)

        if position is None or rotation is None or rationale is None:
            errors.append(f"All motifs must have a position, rotation and rationale. Missing data for '{motif_id}'")
            continue

        if not isinstance(position, list) or len(position) != 2:
            errors.append(f"Invalid position format for '{motif_id}'. Expected [x, y], got {position}")
            continue

        if not isinstance(rotation, (int, float)):
            errors.append(f"All motifs must have a valid rotation. Invalid rotation value for '{motif_id}': {rotation}")

        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"All motifs must have a valid rationale. Invalid or empty rationale for '{motif_id}'")

        if not isinstance(ignore_collision, bool):
            errors.append(f"All motifs must have a valid ignore_collision. Invalid ignore_collision value for '{motif_id}': {ignore_collision}")
        if not isinstance(wall_alignment, bool):
            errors.append(f"All motifs must have a valid wall_alignment. Invalid wall_alignment value for '{motif_id}': {wall_alignment}")

    if errors:
        return False, "\n".join(errors), 3

    return True, "", -1

def validate_arrangement_smc(
    response: str,
    valid_furniture_ids: list,
    new_objects: List[ObjectSpec],
    full_objects: Optional[List[ObjectSpec]] = None,
    existing_motifs_ids: Optional[list] = None,
    enforce_same_layer: bool = False,
    enforce_same_surface: bool = False,
) -> Tuple[bool, str, int]:
    """
    Validate the arrangement response format and content.

    Args:
        response: The JSON string response containing furniture arrangements.
        valid_furniture_ids: List of valid furniture IDs.
        new_objects: The new objects to validate against.
        full_objects: The full list of objects to check for ID conflicts.
        existing_motifs_ids: List of existing motifs ids to avoid duplicate motifs.
        enforce_same_layer: Flag to enforce layer consistency.
        enforce_same_surface: Flag to enforce surface consistency.

    Returns:
        Tuple of (is_valid, error_message, error_code).
        Error codes: -1=success, 1=missing/invalid keys, 3=invalid composition,
        4=invalid furniture, 5=invalid dimensions, 7=json error, 8=unexpected error,
        9=id conflict, 10=duplicate motif, 11=layer mismatch, 12=surface mismatch.
    """
    layout_data, parse_result = _parse_json_response(response)
    if not parse_result[0]:
        return parse_result[1], parse_result[2], 7 if "JSON" in parse_result[1] else 8

    # Check for ID conflicts with existing motifs
    if existing_motifs_ids is not None:
        for motif in layout_data["arrangements"]:
            if motif["id"] in existing_motifs_ids:
                return False, f"ID conflict with existing motifs: {motif['id']}", 10

    # Check for ID conflicts if full_objects is provided
    if full_objects is not None:
        new_object_ids = set(obj.id for obj in new_objects)
        full_object_ids = set(obj.id for obj in full_objects) - new_object_ids
        conflicts = new_object_ids.intersection(full_object_ids)
        if conflicts:
            return False, f"ID conflict between new and full objects: {conflicts}", 9

    # Count occurrences of "arrangements" key in layout_data
    arrangements_count = sum(1 for key in layout_data.keys() if key == "arrangements")
    if arrangements_count > 1:
        return False, "Multiple 'arrangements' keys found in response", 1
    elif arrangements_count == 0:
        return False, "Missing 'arrangements' key in response", 1

    arrangements = layout_data["arrangements"]
    if not isinstance(arrangements, list):
        return False, "'arrangements' must be a list", 1

    # Track used furniture IDs to ensure all are included
    used_furniture_ids = set()
    furniture_usage = {}

    # Build lookup tables for fast access
    id_to_layer = {obj.id: getattr(obj, "placement_layer", None) for obj in new_objects}
    id_to_surface = {obj.id: getattr(obj, "placement_surface", None) for obj in new_objects}

    for arr in arrangements:
        required_keys = {"id", "area_name", "composition", "rationale"}
        if not all(key in arr for key in required_keys):
            return False, f"Missing required keys in arrangement: {required_keys}", 1

        comp = arr["composition"]
        required_comp_keys = {"description", "furniture"}
        if not all(key in comp for key in required_comp_keys):
            return False, f"Missing required keys in composition: {required_comp_keys}", 3

        furniture_entries = comp["furniture"]
        if not isinstance(furniture_entries, list):
            return False, "'furniture' must be a list", 4

        if not furniture_entries:
            return False, "Furniture list cannot be empty", 4

        for entry in furniture_entries:
            if not isinstance(entry, dict):
                return False, "Each furniture entry must be a dict", 4
            if "id" not in entry:
                return False, "Furniture entry missing 'id'", 4

            furniture_id = entry["id"]
            if not isinstance(furniture_id, int):
                return False, f"Furniture 'id' must be an integer, got {type(furniture_id)}", 4
            if furniture_id not in valid_furniture_ids:
                return False, f"Furniture ID {furniture_id} is not in valid IDs: {valid_furniture_ids}", 4

            used_furniture_ids.add(furniture_id)
            amount = entry.get("amount", 1)

            if furniture_id not in furniture_usage:
                furniture_usage[furniture_id] = amount
            else:
                furniture_usage[furniture_id] += amount

            obj_spec = next((obj for obj in new_objects if obj.id == furniture_id), None)
            if obj_spec and furniture_usage[furniture_id] > obj_spec.amount:
                return False, f"Furniture ID {furniture_id} is used more times than allowed. Used: {furniture_usage[furniture_id]}, Allowed: {obj_spec.amount}", 4

        if enforce_same_layer or enforce_same_surface:
            arrangement_layer_set = set()
            arrangement_surface_set = set()
            layer_details = {}
            surface_details = {}

            for entry in furniture_entries:
                f_id = entry["id"]
                layer_val = id_to_layer.get(f_id)
                surf_val = id_to_surface.get(f_id)
                arrangement_layer_set.add(layer_val)
                arrangement_surface_set.add(surf_val)
                layer_details[f_id] = layer_val
                surface_details[f_id] = surf_val

            if enforce_same_layer:
                if None in arrangement_layer_set:
                    offending_ids = [fid for fid, lyr in layer_details.items() if lyr is None]
                    return False, (
                        f"Layer consistency enforcement failed in arrangement '{arr['id']}': "
                        f"objects {offending_ids} do not have an assigned placement_layer."), 11

                if len(arrangement_layer_set) > 1:
                    mapping_str = ", ".join([f"{fid}: {lyr}" for fid, lyr in layer_details.items()])
                    return False, (
                        f"Layer consistency enforcement failed in arrangement '{arr['id']}': "
                        f"objects originate from multiple layers. id->layer mapping: {mapping_str}"), 11

            if enforce_same_surface:
                if None in arrangement_surface_set:
                    offending_ids = [fid for fid, srf in surface_details.items() if srf is None]
                    return False, (
                        f"Surface consistency enforcement failed in arrangement '{arr['id']}': "
                        f"objects {offending_ids} do not have an assigned placement_surface."), 12

                if len(arrangement_surface_set) > 1:
                    mapping_str = ", ".join([f"{fid}: {srf}" for fid, srf in surface_details.items()])
                    return False, (
                        f"Surface consistency enforcement failed in arrangement '{arr['id']}': "
                        f"objects originate from multiple surfaces. id->surface mapping: {mapping_str}"), 12

    if used_furniture_ids != set(valid_furniture_ids):
        missing_ids = set(valid_furniture_ids) - used_furniture_ids
        return False, f"Not all furniture IDs were used. Missing: {missing_ids}", 4

    return True, "", -1

def validate_layered_layout(response: str, layer_data: Dict[str, Any], large_object_names: List[str],
                            valid_ids: List[str] = [], target_layer: str = "", motif_object_names: List[str] = []) -> Tuple[bool, str, int]:
    """
    Validate the layered layout response against the provided layer data.

    Args:
        layer_data: Original layer data with surfaces
        response: JSON response string to validate
        large_object_names: List of valid large object instances
        valid_ids: List of valid motif IDs to check against
        target_layer: Specific layer to target for validation
        motif_object_names: List of valid motif object names to check against

    Returns:
        Tuple of (is_valid, error_message, error_code).
        Error codes: -1=success, 1=json error, 2=layer mismatch, 3=surface mismatch,
        4=invalid placement, 5=invalid id.
    """
    layout, parse_result = _parse_json_response(response)
    if not parse_result[0]:
        return parse_result[1], parse_result[2], 1

    used_ids = set()

    for large_obj_name, obj_layout in layout.items():
        base_name = next((key for key in layer_data.keys() if large_obj_name.startswith(key)), None)

        if not base_name:
            if large_obj_name not in large_object_names:
                available_bases = list(layer_data.keys())
                return False, f"Invalid object name: {large_obj_name}. Expected either exact match from {large_object_names} or instance name starting with one of: {available_bases}", 2
            base_name = large_obj_name

        object_layers = layer_data[base_name]

        if not any(key.startswith("layer_") for key in obj_layout.keys()):
            return False, f"No layer data found for {large_obj_name}", 2

        if target_layer:
            layer_keys = [key for key in obj_layout.keys() if key.startswith("layer_")]
            if target_layer not in layer_keys:
                return False, f"Target layer {target_layer} not found in response for {large_obj_name}", 2
            if len(layer_keys) > 1:
                return False, f"When targeting layer {target_layer}, only that layer should be present, found: {layer_keys}", 2

        for layer_id, layer_content in obj_layout.items():
            if not layer_id.startswith("layer_"):
                continue

            if target_layer and layer_id != target_layer:
                continue

            if layer_id not in object_layers:
                return False, f"Invalid layer {layer_id} for {large_obj_name}. Expected layers: {set(object_layers.keys())}", 2

            original_surfaces = {f"surface_{i}": surface for i, surface in
                                enumerate(object_layers[layer_id]['surfaces'])}

            if set(layer_content.keys()) != set(original_surfaces.keys()):
                expected_surfaces = set(original_surfaces.keys())
                got_surfaces = set(layer_content.keys())
                missing_surfaces = expected_surfaces - got_surfaces
                extra_surfaces = got_surfaces - expected_surfaces

                error_msg = f"Surface mismatch in {large_obj_name} {layer_id}."
                if missing_surfaces:
                    error_msg += f" Missing surfaces: {sorted(missing_surfaces)} (include as empty arrays [] if no objects)."
                if extra_surfaces:
                    error_msg += f" Unexpected surfaces: {sorted(extra_surfaces)}."
                error_msg += f" Expected exactly: {sorted(expected_surfaces)}"

                return False, error_msg, 3

            for surface_id, objects in layer_content.items():
                if not isinstance(objects, list):
                    return False, f"Objects for {surface_id} in {large_obj_name} must be a list", 4

                surface_data = original_surfaces[surface_id]
                if not objects:
                    continue

                for obj in objects:
                    required_props = ['position', 'rotation', 'rationale']
                    if not all(key in obj for key in required_props):
                        return False, f"Missing required properties in object for {large_obj_name}: {obj}. Required: {required_props}", 4

                    if not isinstance(obj['position'], list) or len(obj['position']) != 2:
                        return False, f"Invalid position format in {large_obj_name}: {obj['position']}", 4

                    if not isinstance(obj['rotation'], dict) or ('angle' not in obj['rotation'] and 'facing' not in obj['rotation']):
                        return False, f"Invalid rotation format in {large_obj_name}: {obj['rotation']}. Must contain either 'angle' or 'facing'", 4

                    if not isinstance(obj['rationale'], str) or not obj['rationale'].strip():
                        return False, f"Invalid rationale in {large_obj_name}", 4

                    if 'facing' in obj['rotation']:
                        if not isinstance(obj['rotation']['facing'], str) or not obj['rotation']['facing'].strip():
                            return False, f"Invalid facing value in rotation for {large_obj_name}", 4
                        if motif_object_names is not None and obj['rotation']['facing'] not in motif_object_names:
                            return False, f"Invalid facing ID in rotation: '{obj['rotation']['facing']}'. Must reference an object within the same motif. Available objects: {sorted(motif_object_names)}", 5

                    if 'id' in obj:
                        if obj['id'] not in valid_ids:
                            return False, f"Invalid ID: {obj['id']}. Expected one of: {valid_ids}", 5
                        used_ids.add(obj['id'])

    if valid_ids is not None and used_ids != set(valid_ids):
        missing_ids = set(valid_ids) - used_ids
        return False, f"Not all valid IDs were used. Missing: {missing_ids}", 5

    return True, "", -1

def validate_floorplan(response: str) -> Tuple[bool, str, int]:
    """
    Validate the floorplan response format and content.

    Args:
        response: The JSON response containing floorplan and door location

    Returns:
        Tuple of (is_valid, error_message, error_code).
        Error codes: -1=success, 1=missing keys, 2=invalid floorplan format,
        3=invalid door format, 4=invalid geometry, 5=invalid door placement, 6=error.
    """
    layout_data, parse_result = _parse_json_response(response)
    if not parse_result[0]:
        return parse_result[1], parse_result[2], 6

    if not all(key in layout_data for key in ["room_vertices", "door_location"]):
        return False, "Missing required keys: floorplan and door_location", 1

    room_vertices = layout_data["room_vertices"]
    door_location = layout_data["door_location"]

    if not isinstance(room_vertices, list):
        return False, "Floorplan must be a list of coordinates", 2

    for vertex in room_vertices:
        if not isinstance(vertex, list) or len(vertex) != 2:
            return False, f"Invalid vertex format: {vertex}. Expected [x,y] list", 2
        if not all(isinstance(coord, (int, float)) for coord in vertex):
            return False, f"Invalid coordinate types in vertex: {vertex}", 2

    if not isinstance(door_location, list) or len(door_location) != 2:
        return False, "Door location must be a list of [x,y] coordinates", 3
    if not all(isinstance(coord, (int, float)) for coord in door_location):
        return False, f"Invalid coordinate types in door location: {door_location}", 3

    if len(room_vertices) < 3:
        return False, "Floorplan must have at least 3 vertices", 4

    if room_vertices[0] != [0, 0]:
        return False, "First vertex must be at [0,0]", 4

    try:
        poly = Polygon(room_vertices)
        if not poly.is_valid or poly.area <= 0:
            return False, "Invalid polygon geometry or zero area", 4
    except Exception as e:
        return False, f"Invalid polygon geometry: {str(e)}", 4

    door_point = Point(door_location)
    if not door_point.distance(poly.exterior) < 0.01:
        return False, "Door must be placed on the room perimeter", 5

    return True, "", -1

def validate_room_type(response: str) -> Tuple[bool, str, int]:
    """
    Validate the room type response format and content.

    Args:
        response: The JSON response containing room type

    Returns:
        Tuple of (is_valid, error_message, error_code).
        Error codes: -1=success, 1=json error, 2=missing key, 3=invalid value, 4=error.
    """
    data, parse_result = _parse_json_response(response)
    if not parse_result[0]:
        return parse_result[1], parse_result[2], 1

    if "room_type" not in data:
        return False, "Missing 'room_type' key in response", 2

    room_type = data["room_type"]
    if not isinstance(room_type, str) or not room_type.strip():
        return False, "Room type must be a non-empty string", 3

    return True, "", -1

def validate_wall_position(response: str, valid_motif_ids: list) -> Tuple[bool, str, int]:
    """
    Validate the wall position response from the VLM.

    Args:
        response: The JSON string response from the VLM.
        valid_motif_ids: List of valid motif IDs.

    Returns:
        Tuple of (is_valid, error_message, error_code).
        Error codes: -1=success, 1=missing positions key, 2=invalid format,
        3=validation errors, 4=json error, 5=key error, 6=unexpected error.
    """
    position_data, parse_result = _parse_json_response(response)
    if not parse_result[0]:
        return parse_result[1], parse_result[2], 4

    if "positions" not in position_data:
        return False, "Missing 'positions' key in the response.", 1

    positions = position_data["positions"]
    if not isinstance(positions, list):
        return False, "'positions' must be a list.", 2

    used_motif_ids = set()
    errors = []

    for item in positions:
        required_fields = {"id", "position", "rationale"}
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            errors.append(f"Position entry missing required fields: {', '.join(missing_fields)}")
            continue

        motif_id = item["id"]
        if motif_id not in valid_motif_ids:
            errors.append(f"Invalid motif id: '{motif_id}'")
        else:
            used_motif_ids.add(motif_id)

        position = item["position"]
        if not isinstance(position, list) or len(position) != 2:
            errors.append(f"Position for motif '{motif_id}' must be a 2D array [x, y]")
        elif not all(isinstance(coord, (int, float)) for coord in position):
            errors.append(f"Position coordinates for motif '{motif_id}' must be numbers")

        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"Rationale for motif '{motif_id}' must be a non-empty string")

    missing_motifs = set(valid_motif_ids) - used_motif_ids
    if missing_motifs:
        errors.append(f"Not all valid motifs were positioned. Missing: {', '.join(missing_motifs)}")

    if errors:
        return False, "\n".join(errors), 3

    return True, "", -1

def validate_wall_objects(response: str, wall_object_ids: list, valid_wall_ids: list) -> Tuple[bool, str, int]:
    """
    Validate the wall objects response from the VLM.

    Args:
        response: The JSON string response from the VLM.
        wall_object_ids: List of wall object IDs.
        valid_wall_ids: List of valid wall IDs.

    Returns:
        Tuple of (is_valid, error_message, error_code).
        Error codes: -1=success, 1=missing objects key, 2=invalid format, 3=validation errors, 4=json error.
    """
    objects_data, parse_result = _parse_json_response(response)
    if not parse_result[0]:
        return parse_result[1], parse_result[2], 4

    if "objects" not in objects_data:
        return False, "Missing 'objects' key in the response.", 1

    objects = objects_data["objects"]
    if not isinstance(objects, list):
        return False, "'objects' must be a list.", 2

    errors = []
    for obj in objects:
        required_fields = {"id", "wall_id", "rationale"}
        missing_fields = required_fields - set(obj.keys())
        if missing_fields:
            errors.append(f"Object entry missing required fields: {', '.join(missing_fields)}")
            continue

        obj_id = obj["id"]
        if obj_id not in wall_object_ids:
            errors.append(f"Invalid object id: '{obj_id}'")

        wall_id = obj["wall_id"]
        if wall_id not in valid_wall_ids:
            errors.append(f"Invalid wall_id: '{wall_id}'")

    if errors:
        return False, "\n".join(errors), 3

    return True, "", -1

def _detect_duplicate_json_keys(json_str: str) -> List[str]:
    """
    Detect duplicate keys that occur **within the same JSON object**. This
    leverages ``json.loads`` with a custom ``object_pairs_hook`` so that
    duplicate *sibling* keys are reported accurately.  Keys with the same
    name but belonging to **different parent objects are allowed** (e.g.
    ``{"nightstand": {"layer_0": ...}, "nightstand_2": {"layer_0": ...}}``
    should **not** be considered a duplicate).

    The previous heuristic (regex over the raw string) produced false
    positives when the same key appeared under different parents. Using the
    hook ensures we only flag duplicates that violate JSON semantics within
    a single object.

    Returns:
        List[str]: Human-readable descriptions of duplicate keys detected.
    """
    extracted_json: str
    try:
        extracted_json = extract_json(json_str)
    except Exception:
        # If we cannot even isolate JSON, defer error handling to the caller.
        return ["Could not extract JSON from response"]

    duplicate_keys: list[str] = []

    def _hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Custom ``object_pairs_hook`` that records duplicate keys."""
        local_counter: Counter[str] = Counter(key for key, _ in pairs)
        for key, cnt in local_counter.items():
            if cnt > 1:
                duplicate_keys.append(key)
        # Convert to dict as the default behaviour.
        return {k: v for k, v in pairs}

    try:
        json.loads(extracted_json, object_pairs_hook=_hook)
    except json.JSONDecodeError:
        # Errors here will be handled by the caller; keep duplicate list empty.
        return []

    # Craft readable messages
    return [f"Duplicate key '{k}' found multiple times within the same object" for k in duplicate_keys]

def validate_small_object_response(
    response_str: str,
    expected_small_specs: List[ObjectSpec], # Defines total expected objects and their amounts
    parent_names: List[str], # Exact parent instance names expected as keys in the response
    layer_data: Dict[str, Any]  # Defines expected layer/surface structure under each parent
) -> Tuple[bool, str, int]:
    """
    Validates the VLM's layered small object placement response for structural integrity,
    individual object validity, and overall quantitative completeness against expected specs.

    Args:
        response_str (str): The JSON string response from the VLM.
        expected_small_specs (List[ObjectSpec]): List of ObjectSpecs that define the total
                                                 amount of each small object type expected to be placed.
        parent_names (List[str]): The exact list of parent object instance names expected as
                                  top-level keys in the VLM's JSON response.
        layer_data (Dict[str, Any]): The layer/surface structure expected under each parent key,
                                     e.g., {"shelf_0": {"layer_0": {"surfaces": [{}, {}]}}}.

    Returns:
        Tuple[bool, str, int]: (isValid, errorMessage, errorCode)
        Error Codes:
            -1: Success
             4: Invalid JSON format or could not extract JSON from response.
             5: Unexpected error during validation (internal).
             6: Duplicate JSON keys detected (malformed structure).
             7: Parent contains unexpected layer(s).
             8: Layer contains unexpected surface(s).
             9: Objects for a surface is not a list.
            10: An object entry in the list is not a dictionary.
            11: Invalid or missing 'name' for an object (e.g., not string, empty).
            12: Invalid or missing 'dimensions' for an object (e.g., not list of 3 numbers).
            13: Invalid or missing 'amount' for an object (e.g., not positive integer).
            14: Data for an expected parent object is not a dictionary.
            15: Data for an expected layer is not a dictionary.
            16: Missing surface(s) in an expected layer.
            20: Completeness mismatch (VLM placed unexpected, not enough, or too many objects in total quantities).
            21: Invalid ObjectSpec found in `expected_small_specs` (internal data error).
    """
    # Early detection of duplicate JSON keys
    duplicate_key_errors = _detect_duplicate_json_keys(response_str)
    if duplicate_key_errors:
        error_msg = "JSON structure validation failed - duplicate keys detected:\n" + "\n".join(duplicate_key_errors)
        error_msg += "\n\nEach layer (e.g., 'layer_0', 'layer_1') must appear exactly once under each parent object."
        error_msg += "\nEach surface (e.g., 'surface_0', 'surface_1') must appear exactly once under each layer."
        return False, error_msg, 6
    
    try:
        parsed_data = json.loads(extract_json(response_str))
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format in response: {str(e)}", 4

    # For completeness check
    llm_placed_counts: Counter[Tuple[str, Tuple[float, ...]]] = Counter()

    try:
        # 1. Validate parent keys (allow missing keys)
        actual_parent_keys_from_response = set(parsed_data.keys())
        expected_parent_keys_as_per_arg = set(parent_names)

        extra_parents = actual_parent_keys_from_response - expected_parent_keys_as_per_arg
        if extra_parents:
            error_msg = (
                f"Parent key mismatch: Unexpected parent key(s) found in response: {list(extra_parents)}. "
                f"All keys must be within the expected list: {parent_names}"
            )
            return False, error_msg, 1

        # 2. Validate layers and surfaces for each parent present in the response
        for parent_key in actual_parent_keys_from_response:
            response_parent_data = parsed_data[parent_key]
            if not isinstance(response_parent_data, dict):
                return False, f"Data for parent '{parent_key}' is not a dictionary.", 14

            # layer_data keys are base names (e.g. "shelf"), parent_key is instance name (e.g. "shelf_0")
            # We need to find the base name from parent_key to look up in layer_data.
            # This assumes parent_key (instance name) starts with the base name.
            # A more robust mapping might be needed if names don't follow this convention.
            base_parent_name_for_layer_lookup = next((bn for bn in layer_data if parent_key.startswith(bn)), None)
            if not base_parent_name_for_layer_lookup:
                 # This implies that a parent_key from `parent_names` (which are instance names from VLM response)
                 # does not have a corresponding base definition in `layer_data`.
                 # This could be a setup issue with `layer_data` or `parent_names`.
                return False, f"Could not find layer definition base for parent instance '{parent_key}' in layer_data keys: {list(layer_data.keys())}.", 5 

            expected_layers_for_parent = layer_data.get(base_parent_name_for_layer_lookup)
            if expected_layers_for_parent is None:
                if response_parent_data: # If response has layers but none expected
                    return False, f"Parent '{parent_key}' (base: '{base_parent_name_for_layer_lookup}') has data in response, but no layers are defined for its base type in layer_data.", 7
                continue # Correctly no layers in response, and none expected for this parent's base type.
            
            actual_layer_keys = set(response_parent_data.keys())
            expected_layer_keys_set = set(expected_layers_for_parent.keys())

            missing_layers = expected_layer_keys_set - actual_layer_keys
            if missing_layers:
                return False, f"Parent '{parent_key}' (base: '{base_parent_name_for_layer_lookup}'): Missing layer(s) {list(missing_layers)}.", 2

            extra_layers = actual_layer_keys - expected_layer_keys_set
            if extra_layers:
                return False, f"Parent '{parent_key}' (base: '{base_parent_name_for_layer_lookup}'): Contains unexpected layer(s) {list(extra_layers)}.", 7
            
            for layer_key, expected_layer_info in expected_layers_for_parent.items():
                response_layer_data = response_parent_data[layer_key] # Known to exist due to checks above
                if not isinstance(response_layer_data, dict):
                    return False, f"Data for layer '{layer_key}' in parent '{parent_key}' is not a dictionary.", 15

                num_expected_surfaces = len(expected_layer_info.get("surfaces", []))
                expected_surface_keys = {f"surface_{j}" for j in range(num_expected_surfaces)}
                actual_surface_keys_in_layer = set(response_layer_data.keys())

                missing_surfaces = expected_surface_keys - actual_surface_keys_in_layer
                if missing_surfaces:
                    return False, f"Parent '{parent_key}', Layer '{layer_key}': Missing surface(s) {list(missing_surfaces)}.", 16

                extra_surfaces = actual_surface_keys_in_layer - expected_surface_keys
                if extra_surfaces:
                    return False, f"Parent '{parent_key}', Layer '{layer_key}': Contains unexpected surface(s) {list(extra_surfaces)}.", 8
                
                for surface_key in expected_surface_keys: # These surfaces are now confirmed to exist
                    objects_on_surface = response_layer_data[surface_key]
                    if not isinstance(objects_on_surface, list):
                        return False, f"Objects for parent '{parent_key}', layer '{layer_key}', surface '{surface_key}' must be a list, got {type(objects_on_surface).__name__}.", 9
                    
                    # 3. Validate individual object properties and aggregate for completeness
                    for obj_idx, obj_data in enumerate(objects_on_surface):
                        obj_path_str = f"Parent '{parent_key}', Layer '{layer_key}', Surface '{surface_key}', Object at index {obj_idx}"
                        if not isinstance(obj_data, dict):
                            return False, f"{obj_path_str}: Entry is not a dictionary.", 10

                        obj_name = obj_data.get("name")
                        obj_dims_list = obj_data.get("dimensions")
                        obj_amount = obj_data.get("amount")

                        if not isinstance(obj_name, str) or not obj_name:
                            return False, f"{obj_path_str}: Invalid or missing 'name' (must be a non-empty string). Got: '{obj_name}'", 11
                        if not (isinstance(obj_dims_list, list) and len(obj_dims_list) == 3 and all(isinstance(d, (int, float)) for d in obj_dims_list)):
                            return False, f"{obj_path_str} (name: '{obj_name}'): Invalid 'dimensions' (must be list of 3 numbers). Got: {obj_dims_list}", 12
                        if any(d <= 0 for d in obj_dims_list):
                            return False, f"{obj_path_str} (name: '{obj_name}'): Invalid 'dimensions' (all dimension values must be greater than 0). Got: {obj_dims_list}", 12
                        if not isinstance(obj_amount, int) or obj_amount <= 0:
                            return False, f"{obj_path_str} (name: '{obj_name}'): Invalid 'amount' (must be a positive integer greater than 0). Got: {obj_amount}", 13
                        
                        # Aggregate for completeness check
                        dims_tuple = tuple(float(d) for d in obj_dims_list)
                        llm_placed_counts[(obj_name, dims_tuple)] += obj_amount
        
        # 4. Aggregate counts from expected_small_specs (total expected amounts)
        expected_total_counts: Counter[Tuple[str, Tuple[float, ...]]] = Counter()
        for spec in expected_small_specs:
            if not (isinstance(spec.name, str) and spec.name and \
                    isinstance(spec.dimensions, list) and len(spec.dimensions) == 3 and \
                    all(isinstance(d, (int, float)) for d in spec.dimensions) and \
                    isinstance(spec.amount, int) and spec.amount > 0):
                # This is an issue with the input data provided to the validator.
                return False, f"Internal validation setup error: Invalid ObjectSpec encountered in expected_small_specs. ID: {getattr(spec, 'id', 'N/A')}, Name: {getattr(spec, 'name', 'N/A')}", 21
            
            spec_dims_tuple = tuple(float(d) for d in spec.dimensions)
            expected_total_counts[(spec.name, spec_dims_tuple)] += spec.amount

        # 5. Compare aggregated VLM placements with total expected counts
        completeness_error_messages = []
        all_object_keys = set(llm_placed_counts.keys()) | set(expected_total_counts.keys())

        for key in sorted(list(all_object_keys)):
            obj_name, obj_dims_tuple = key
            llm_total_amount = llm_placed_counts.get(key, 0)
            expected_total_amount = expected_total_counts.get(key, 0)

            if llm_total_amount < expected_total_amount:
                completeness_error_messages.append(f"Object '{obj_name}' (dims: {obj_dims_tuple}): Expected total {expected_total_amount}, VLM placed {llm_total_amount} (Not enough).")
            elif llm_total_amount > expected_total_amount:
                completeness_error_messages.append(f"Object '{obj_name}' (dims: {obj_dims_tuple}): Expected total {expected_total_amount}, VLM placed {llm_total_amount} (Too many).")
            # If an object is in llm_placed_counts but not expected_total_counts (i.e. expected_total_amount is 0), it implies an unexpected object.
            # This is covered by the llm_total_amount > expected_total_amount check when expected_total_amount is 0.

        if completeness_error_messages:
            # Add helpful context about available surfaces for better debugging
            total_surfaces = 0
            surface_breakdown = []
            for parent_name in parent_names:
                base_name = next((bn for bn in layer_data if parent_name.startswith(bn)), None)
                if base_name and base_name in layer_data:
                    parent_layers = layer_data[base_name]
                    for layer_key, layer_info in parent_layers.items():
                        if layer_key.startswith('layer_'):
                            num_surfaces = len(layer_info.get("surfaces", []))
                            total_surfaces += num_surfaces
                            surface_breakdown.append(f"{parent_name}.{layer_key}: {num_surfaces} surfaces")
            
            error_msg = "Small object placement completeness validation failed:\n" + "\n".join(completeness_error_messages)
            error_msg += f"\n\nAvailable surfaces for distribution ({total_surfaces} total):\n" + "\n".join(surface_breakdown)
            error_msg += "\n\nHint: Distribute objects across multiple layers/surfaces to reach exact totals."
            return False, error_msg, 20

        return True, "", -1 # All validations passed

    except Exception as e:
        # Catch-all for unexpected errors during the main validation logic
        return False, f"Unexpected error during validation: {type(e).__name__} - {e} (Line: {e.__traceback__.tb_lineno if e.__traceback__ else 'N/A'})", 5

