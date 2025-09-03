from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Any
import json
import re
from pathlib import Path

import sys
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from hsm_core.scene.specifications.object_spec import ObjectSpec

@dataclass
class SceneSpec:
    @staticmethod
    def sanitize_name(name: str) -> str:
        # Replace hyphens and spaces with underscores
        return re.sub(r'[-\s]+', '_', name.strip())
    
    large_objects: List[ObjectSpec]
    small_objects: List[ObjectSpec]
    wall_objects: List[ObjectSpec]
    ceiling_objects: List[ObjectSpec]
    # Track next available ID for each type
    _next_large_id: int = field(default=1, repr=False)
    _next_small_id: int = field(default=1000, repr=False)
    _next_wall_id: int = field(default=2000, repr=False)
    _next_ceiling_id: int = field(default=3000, repr=False)
    _id_to_object_map: dict[int, ObjectSpec] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize ID counters and build the ID-to-object map."""
        if self.large_objects:
            self._next_large_id = max(self._next_large_id, max(obj.id for obj in self.large_objects) + 1)
        else:
            self._next_large_id = max(self._next_large_id, 1)
        if self.small_objects:
            self._next_small_id = max(self._next_small_id, max(1000, max((obj.id for obj in self.small_objects), default=999) + 1))
        else:
            self._next_small_id = max(self._next_small_id, 1000)
        if self.wall_objects:
            self._next_wall_id = max(self._next_wall_id, max(2000, max((obj.id for obj in self.wall_objects), default=1999) + 1))
        else:
            self._next_wall_id = max(self._next_wall_id, 2000)
        if self.ceiling_objects:
            self._next_ceiling_id = max(self._next_ceiling_id, max(3000, max((obj.id for obj in self.ceiling_objects), default=2999) + 1))
        else:
            self._next_ceiling_id = max(self._next_ceiling_id, 3000)
        # Build the ID-to-object map
        self._id_to_object_map = {obj.id: obj for obj in self.large_objects + self.small_objects + self.wall_objects + self.ceiling_objects}

    @property
    def layered_small_objects(self) -> dict[int, dict[str, dict[int, list[ObjectSpec]]]]:
        """Hierarchical structure for layer-organized small objects, always computed from small_objects."""
        layered: dict[int, dict[str, dict[int, list[ObjectSpec]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for obj in self.small_objects:
            if obj.parent_object is not None and obj.placement_layer is not None and obj.placement_surface is not None:
                layered[obj.parent_object][obj.placement_layer][obj.placement_surface].append(obj)
        return layered

    def get_object_by_id(self, obj_id: int) -> Optional[ObjectSpec]:
        """Find object by ID in all objects."""
        result = self._id_to_object_map.get(obj_id)
        if result is None:
            # Debug: If not found in map, check if it exists in the lists but map is stale
            all_objects = self.large_objects + self.small_objects + self.wall_objects + self.ceiling_objects
            for obj in all_objects:
                if obj.id == obj_id:
                    print(f"DEBUG: Object {obj_id} found in lists but not in _id_to_object_map. Updating map.")
                    self._id_to_object_map[obj_id] = obj
                    return obj
        return result

    @classmethod
    def from_json(cls, json_str: str, required: bool = False) -> 'SceneSpec':
        """Create SceneSpec from JSON string, remapping IDs and managing parent/required flags.

        Args:
            json_str: JSON string representing the scene spec.
            required: If True, set all objects' required flag to True.

        Returns:
            SceneSpec: The constructed scene spec with correct IDs and flags.
        """
        data = json.loads(json_str)

        def parse_objects(obj_data_list: Optional[list[dict[str, Any]]]) -> list[ObjectSpec]:
            parsed_list: list[ObjectSpec] = []
            if not obj_data_list:
                return parsed_list
            for raw_obj_data in obj_data_list:
                try:
                    raw_obj_data['name'] = cls.sanitize_name(raw_obj_data['name'])
                    spec = ObjectSpec(**raw_obj_data)
                    parsed_list.append(spec)
                except TypeError as e:
                    print(f"Warning: Skipping object due to missing data or type mismatch in JSON: {raw_obj_data}. Error: {e}", file=sys.stderr)
            return parsed_list

        # Parse all objects from JSON
        large_objs_raw = parse_objects(data.get("objects"))
        wall_objs_raw = parse_objects(data.get("wall_objects"))
        ceiling_objs_raw = parse_objects(data.get("ceiling_objects"))
        small_objs_raw = parse_objects(data.get("small_objects"))

        # Remap IDs for uniqueness and correct starting points
        id_maps = {"large": {}, "small": {}, "wall": {}, "ceiling": {}}
        next_ids = {"large": 1, "small": 1000, "wall": 2000, "ceiling": 3000}

        def remap_objs(objs: list[ObjectSpec], obj_type: str) -> list[ObjectSpec]:
            remapped = []
            for obj in objs:
                new_id = next_ids[obj_type]
                id_maps[obj_type][obj.id] = new_id
                next_ids[obj_type] += 1
                # Copy and update required if needed
                remapped.append(ObjectSpec(
                    id=new_id,
                    name=obj.name.lower(), # Convert to lowercase
                    description=obj.description,
                    dimensions=list(obj.dimensions),
                    amount=obj.amount,
                    parent_object=obj.parent_object,  # Will update for smalls later
                    placement_layer=obj.placement_layer,
                    placement_surface=obj.placement_surface,
                    wall_id=obj.wall_id,
                    required=required if required else obj.required,
                    is_parent=False  # Will update later
                ))
            return remapped

        large_objects = remap_objs(large_objs_raw, "large")
        wall_objects = remap_objs(wall_objs_raw, "wall")
        ceiling_objects = remap_objs(ceiling_objs_raw, "ceiling")
        # For smalls, remap parent_object after all IDs are known
        small_objects = []
        for obj in small_objs_raw:
            # Determine new parent ID
            parent_id = obj.parent_object
            new_parent_id = None
            if parent_id is not None:
                # Try all possible object types for parent
                for t in ["large", "wall", "ceiling"]:
                    if parent_id in id_maps[t]:
                        new_parent_id = id_maps[t][parent_id]
                        break
            new_id = next_ids["small"]
            id_maps["small"][obj.id] = new_id
            next_ids["small"] += 1
            small_objects.append(ObjectSpec(
                id=new_id,
                name=obj.name,
                description=obj.description,
                dimensions=list(obj.dimensions),
                amount=obj.amount,
                parent_object=new_parent_id,
                placement_layer=obj.placement_layer,
                placement_surface=obj.placement_surface,
                wall_id=obj.wall_id,
                required=required if required else obj.required,
                is_parent=False  # Will update below
            ))

        # Set is_parent=True for any object referenced as a parent
        parent_id_set = set(obj.parent_object for obj in small_objects if obj.parent_object is not None)
        for obj in large_objects + wall_objects + ceiling_objects:
            if obj.id in parent_id_set:
                obj.is_parent = True
            
        scene_spec = cls(
            large_objects=large_objects,
            small_objects=small_objects,
            wall_objects=wall_objects,
            ceiling_objects=ceiling_objects
        )

        print("Loaded SceneSpec: ", scene_spec.to_dict())
        # pprint.pprint(scene_spec.to_dict())

        return scene_spec

    def to_dict(self) -> Dict[str, Any]:
        """Convert SceneSpec to dictionary format."""
        result = {
            "objects": [obj.to_dict() for obj in self.large_objects],
            "small_objects": [obj.to_dict() for obj in self.small_objects],
            "wall_objects": [obj.to_dict() for obj in self.wall_objects],
            "ceiling_objects": [obj.to_dict() for obj in self.ceiling_objects]
        }
        
        # Add layered small objects in a format suitable for JSON serialization
        layered_dict = {}
        for parent_id, layers in self.layered_small_objects.items():
            if not layers:
                continue
            layered_dict[str(parent_id)] = {}
            for layer_id, surfaces in layers.items():
                if not surfaces:
                    continue
                layered_dict[str(parent_id)][layer_id] = {}
                for surface_id, objects in surfaces.items():
                    if not objects:
                        continue
                    layered_dict[str(parent_id)][layer_id][str(surface_id)] = [obj.to_dict() for obj in objects]
        
        if layered_dict:
            result["layered_small_objects"] = layered_dict
            
        return result

    def to_json(self) -> str:
        """Convert SceneSpec to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filepath: Path | str) -> None:
        """Save scene specification to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with filepath.open('w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, filepath: Path | str) -> 'SceneSpec':
        """Load scene specification from a JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Scene spec file not found: {filepath}")
            
        with filepath.open('r', encoding='utf-8') as f:
            return self.from_json(f.read())

    def _get_all_current_ids(self) -> set[int]:
        all_ids = set()
        for lst in [self.large_objects, self.small_objects, self.wall_objects, self.ceiling_objects]:
            for obj_spec in lst:
                all_ids.add(obj_spec.id)
        return all_ids

    def _generate_new_id_for_type(self, object_type: str, all_existing_ids: set[int]) -> int:
        id_attr_name = f"_next_{object_type}_id"
        current_counter_val = getattr(self, id_attr_name)
        
        next_id_val = current_counter_val
        while next_id_val in all_existing_ids:
            next_id_val += 1
        
        setattr(self, id_attr_name, next_id_val + 1)
        return next_id_val

    def add_objects(self, objects_to_add: List[ObjectSpec], object_type: str) -> 'SceneSpec':
        """
        Add objects to the scene with automatically managed IDs.
        Returns a new SceneSpec with the added objects and updated IDs.
        
        Args:
            objects: List of ObjectSpec objects to add
            type: Object type ("large", "small", "wall", or "ceiling")
            
        Returns:
            A new SceneSpec object containing only the added objects with updated IDs
        """
        if not objects_to_add:
            return SceneSpec(large_objects=[], small_objects=[], wall_objects=[], ceiling_objects=[])
            
        all_current_ids = self._get_all_current_ids()
        id_remap_within_batch = {} 
        processed_objects_for_this_call = []

        for original_obj_spec in objects_to_add:
            # Use original_obj_spec.id if it's unique and desired, otherwise generate new.
            # For this refined version, we will always generate a new ID based on type,
            # ensuring it's globally unique.
            new_id = self._generate_new_id_for_type(object_type, all_current_ids)
            all_current_ids.add(new_id) 
            id_remap_within_batch[original_obj_spec.id] = new_id

            new_spec = ObjectSpec(
                id=new_id,
                name=original_obj_spec.name,
                description=original_obj_spec.description,
                dimensions=original_obj_spec.dimensions.copy() if isinstance(original_obj_spec.dimensions, list) else original_obj_spec.dimensions,
                amount=original_obj_spec.amount,
                parent_object=original_obj_spec.parent_object,
                placement_layer=getattr(original_obj_spec, 'placement_layer', None),
                placement_surface=getattr(original_obj_spec, 'placement_surface', None),
                wall_id=getattr(original_obj_spec, 'wall_id', None)
            )
            processed_objects_for_this_call.append(new_spec)

        for spec_in_batch in processed_objects_for_this_call:
            if spec_in_batch.parent_object is not None and spec_in_batch.parent_object in id_remap_within_batch:
                spec_in_batch.parent_object = id_remap_within_batch[spec_in_batch.parent_object]

        if object_type == "large":
            self.large_objects.extend(processed_objects_for_this_call)
        elif object_type == "small":
            self.small_objects.extend(processed_objects_for_this_call)
            for obj in processed_objects_for_this_call:
                if obj.parent_object is not None and \
                   obj.placement_layer is not None and \
                   obj.placement_surface is not None:
                    layer_id = obj.placement_layer
                    surface_id = obj.placement_surface
                    if obj.parent_object not in self.layered_small_objects:
                        self.layered_small_objects[obj.parent_object] = defaultdict(lambda: defaultdict(list))
                    self.layered_small_objects[obj.parent_object][layer_id][surface_id].append(obj)
        elif object_type == "wall":
            self.wall_objects.extend(processed_objects_for_this_call)
        elif object_type == "ceiling":
            self.ceiling_objects.extend(processed_objects_for_this_call)
        else:
            raise ValueError(f"Invalid object type: {object_type}")
            
        # Update the ID-to-object map with the newly added objects
        for obj in processed_objects_for_this_call:
            self._id_to_object_map[obj.id] = obj
            
        return_spec_kwargs = {
            "large_objects": [], "small_objects": [],
            "wall_objects": [], "ceiling_objects": []
        }
        # Ensure the key matches the list attribute name
        list_attr_name = f"{object_type}_objects"
        if hasattr(self, list_attr_name):
             return_spec_kwargs[list_attr_name] = processed_objects_for_this_call
        else:
            # Fallback or error if type doesn't match known lists
            raise ValueError(f"Cannot create return spec for unknown type: {object_type}")

        return SceneSpec(**return_spec_kwargs)
    
    def add_layered_objects_from_response(self, response: dict, expected_parent_id: int, expected_parent_name_in_llm_response: str) -> 'SceneSpec':
        """
        Process a layered object response from the VLM and add objects to the scene,
        Return a new SceneSpec with the added objects with corrected IDs.
        
        Args:
            response: Dictionary from VLM with layered structure {parent_name: {layer: {surface: objects}}}
            expected_parent_id: The ID of the parent object these small objects belong to.
            expected_parent_name_in_llm_response: The name of the parent object as used by the VLM
                                                 for keying the response.
            
        Returns:
            A new SceneSpec containing only the added objects with corrected IDs, or an empty one if processing fails.
        """
        all_objects = []
        print(f"Debug: add_layered_objects_from_response. Expected parent ID: {expected_parent_id}, Expected VLM parent name: '{expected_parent_name_in_llm_response}'. Response keys: {list(response.keys())}")

        # Temporary Debugging Prints
        print(f"DEBUG SCENESPEC: Inside add_layered_objects_from_response for parent ID {expected_parent_id}, VLM name '{expected_parent_name_in_llm_response}'")
        print(f"DEBUG SCENESPEC: Wall objects IDs: {[obj.id for obj in self.wall_objects]}")
        print(f"DEBUG SCENESPEC: Large objects IDs: {[obj.id for obj in self.large_objects]}")
        print(f"DEBUG SCENESPEC: Small objects IDs: {[obj.id for obj in self.small_objects]}")
        print(f"DEBUG SCENESPEC: Ceiling objects IDs: {[obj.id for obj in self.ceiling_objects]}")
        found_in_debug_check = False
        for obj_type_list_name in ["wall_objects", "large_objects", "small_objects", "ceiling_objects"]:
            obj_list = getattr(self, obj_type_list_name)
            for obj_instance in obj_list:
                if obj_instance.id == expected_parent_id:
                    print(f"DEBUG SCENESPEC: Found object with ID {expected_parent_id} in {obj_type_list_name} during debug scan. Name: {obj_instance.name}, ID type: {type(obj_instance.id)}, Obj ID type: {type(obj_instance.id)}")
                    found_in_debug_check = True
                    break
            if found_in_debug_check:
                break
        if not found_in_debug_check:
            print(f"DEBUG SCENESPEC: Object with ID {expected_parent_id} NOT FOUND during manual debug scan.")
        # End Temporary Debugging Prints

        expected_parent_object = self.get_object_by_id(expected_parent_id)
        if not expected_parent_object:
            print(f"Warning: Expected parent object with ID {expected_parent_id} not found globally. Cannot process response.")
            return self.add_objects([], "small") # Return empty spec from add_objects

        parent_name_from_llm_response_key = None
        layers_to_process = None

        if not response:
            print(f"Warning: VLM response is empty. No small objects to add for parent ID {expected_parent_id} ('{expected_parent_name_in_llm_response}').")
            return self.add_objects([], "small")

        # Try to find the expected_parent_name_in_llm_response (case-insensitive) in the response keys
        found_key = None
        for key_from_llm in response.keys():
            if key_from_llm.lower() == expected_parent_name_in_llm_response.lower():
                found_key = key_from_llm
                break
        
        if found_key:
            parent_name_from_llm_response_key = found_key
            layers_to_process = response[parent_name_from_llm_response_key]
            print(f"Found and processing for VLM key '{parent_name_from_llm_response_key}' (corresponds to expected parent ID {expected_parent_id}, expected VLM name '{expected_parent_name_in_llm_response}') from multi-key VLM response.")
        else:
            print(f"Warning: Expected parent name '{expected_parent_name_in_llm_response}' (for ID {expected_parent_id}) not found directly as a key in VLM response keys: {list(response.keys())}. Response might be malformed or empty for this parent.")
            return self.add_objects([], "small")

        if not layers_to_process:
             # This case should ideally be caught by earlier checks, but as a safeguard:
            print(f"Warning: No layers found to process for parent '{parent_name_from_llm_response_key}' (ID: {expected_parent_id}).")
            return self.add_objects([], "small")

        print(f"Processing small objects for parent '{parent_name_from_llm_response_key}' (ID: {expected_parent_id})")
            
        for layer_id, surfaces in layers_to_process.items():
            for surface_key, objects_on_surface in surfaces.items():
                surface_id = int(surface_key.split("_")[1])
                for obj_data in objects_on_surface:
                    obj_spec = ObjectSpec(
                        id=self._next_small_id,  # Temporary ID, will be updated by add_objects
                        name=obj_data.get("name", "small_object"),
                        description=obj_data.get("description", ""),
                        dimensions=obj_data.get("dimensions", [0.2, 0.2, 0.2]),
                        amount=obj_data.get("amount", 1),
                        parent_object=expected_parent_id, # Use the confirmed/expected parent ID
                        placement_layer=layer_id,
                        placement_surface=surface_id
                    )
                    all_objects.append(obj_spec)
                        
        # Now add all objects at once; add_objects will handle final ID generation for the small objects themselves
        return self.add_objects(all_objects, "small")

    def add_multi_parent_small_objects(self, llm_response_data: Dict[str, Any], parent_name_to_id_map: Dict[str, int]) -> 'SceneSpec':
        """
        Processes a layered small object response from the VLM, potentially containing multiple parent objects,
        and adds all parsed small objects to the scene spec.
        This method assumes llm_response_data has ALREADY BEEN VALIDATED for structure and content.

        Args:
            llm_response_data (Dict[str, Any]): The validated, parsed JSON data from the VLM's response.
                Expected format: {
                    "parent_instance_name_1": { # e.g., "desk_0"
                        "layer_0": {
                            "surface_0": [
                                { "name": "book", "description": "...", "dimensions": [...], "amount": 3 }, ...
                            ], ...
                        }, ...
                    },
                    "parent_instance_name_2": { ... }, ...
                }
            parent_name_to_id_map (Dict[str, int]): Maps parent instance names (from VLM response keys)
                                                    to their actual integer IDs in the SceneSpec.

        Returns:
            SceneSpec: A new SceneSpec instance containing only the newly added small objects with their
                       final, correct IDs. Returns an empty SceneSpec if no objects are to be added.
        """
        all_new_small_specs: List[ObjectSpec] = []

        # No internal validation here; assumes llm_response_data is pre-validated by validate_small_object_response

        for parent_instance_name, layers in llm_response_data.items():
            actual_parent_id = parent_name_to_id_map.get(parent_instance_name)
            if actual_parent_id is None:
                # This case should ideally be caught by pre-validation if parent_names arg to validator was correct.
                # However, as a safeguard during direct calls or if maps are mismatched:
                print(f"Critical Error: Parent instance name '{parent_instance_name}' in pre-validated VLM response not found in parent_name_to_id_map. Skipping this parent.")
                continue

            for layer_key, surfaces in layers.items():
                for surface_key, objects_on_surface in surfaces.items():
                    surface_index = int(surface_key.split('_')[-1]) # Assumes 'surface_X' format, pre-validated

                    for obj_data in objects_on_surface:
                        # Data integrity (name, dims, amount) assumed to be pre-validated
                        new_spec = ObjectSpec(
                            id=0, # Placeholder ID, SceneSpec.add_objects will assign a real one
                            name=obj_data["name"],
                            description=obj_data.get("description", ""), # Description is optional
                            dimensions=obj_data["dimensions"],
                            amount=obj_data["amount"],
                            parent_object=actual_parent_id,
                            placement_layer=layer_key,
                            placement_surface=surface_index,
                            required=False, # Default for VLM-suggested small objects
                            is_parent=False
                        )
                        all_new_small_specs.append(new_spec)
        
        if not all_new_small_specs:
            # This can happen if the VLM response, though structurally valid, contains no objects.
            print("No small objects to add from the (validated) multi-parent response.")
        
        return self.add_objects(all_new_small_specs, "small")

    def __iter__(self) -> Iterator[ObjectSpec]:
        """
        Iterate over all ObjectSpec instances in the SceneSpec.
        
        Example: print all object ids
            for obj in scene_spec:
                print(obj.id)
        """
        yield from self.large_objects
        yield from self.small_objects
        yield from self.wall_objects
        yield from self.ceiling_objects