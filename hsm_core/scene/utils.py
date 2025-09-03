from __future__ import annotations
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any, Optional
import json
import trimesh
import numpy as np
import os
import logging

from hsm_core.scene.objects import SceneObject
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.motif import SceneMotif
from hsm_core.scene.spec import ObjectSpec, SceneSpec
from hsm_core.utils.util import numpy_to_python
from hsm_core.utils.path_utils import to_relative_path, to_absolute_path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hsm_core.scene_motif.core.arrangement import Arrangement

logger = logging.getLogger(__name__)

#gist from: https://gist.github.com/jannismain/e96666ca4f059c3e5bc28abb711b5c92
class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    MAX_WIDTH = 70
    MAX_ITEMS = 10

    def __init__(self, *args, **kwargs):
        if kwargs.get("indent") is None:
            kwargs["indent"] = 4
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            return self._encode_list(o)
        if isinstance(o, dict):
            return self._encode_object(o)
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )

    def _encode_list(self, o):
        if self._put_on_single_line(o):
            return "[" + ", ".join(self.encode(el) for el in o) + "]"
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

    def _encode_object(self, o):
        if not o:
            return "{}"

        # ensure keys are converted to strings
        o = {str(k) if k is not None else "null": v for k, v in o.items()}

        if self.sort_keys:
            o = dict(sorted(o.items(), key=lambda x: x[0]))

        if self._put_on_single_line(o):
            return (
                "{ "
                + ", ".join(
                    f"{self.encode(k)}: {self.encode(el)}" for k, el in o.items()
                )
                + " }"
            )

        self.indentation_level += 1
        output = [
            f"{self.indent_str}{self.encode(k)}: {self.encode(v)}" for k, v in o.items()
        ]
        self.indentation_level -= 1

        return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        return (
            self._primitives_only(o)
            and len(o) <= self.MAX_ITEMS
            and len(str(o)) - 2 <= self.MAX_WIDTH
        )

    def _primitives_only(self, o: list | tuple | dict):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})"
            )

def clean_local_scratch_path(path_str: str, current_scene_dir: Optional[str] = None) -> str:
    """
    Remove /local-scratch/ prefix from paths and dynamically redirect to new location.

    Args:
        path_str: Path string that may contain /local-scratch/ prefix
        current_scene_dir: Current scene directory to redirect paths to

    Returns:
        str: Cleaned path without /local-scratch/ prefix and redirected to new location
    """
    if not path_str:
        return path_str

    # Remove local-scratch prefix and fix incorrect hsm_core/scene paths
    cleaned = path_str.replace("/local-scratch/", "/").replace("hsm_core/scene/", "scene/")
    return cleaned


def _resolve_motif_path(motif_data: Dict[str, Any], key: str, project_root: Path, output_dir: Path, warning_list: List[str]) -> None:
    """Resolve a path in motif_data to relative path with fallback handling."""
    if not motif_data.get(key):
        return

    try:
        motif_data[key] = to_relative_path(motif_data[key], project_root)
    except ValueError:
        try:
            motif_data[key] = to_relative_path(motif_data[key], output_dir)
        except ValueError:
            warning_list.append(f"Could not make {key} path relative: {motif_data[key]}")


def _convert_position_to_3d(pos: List[float]) -> Tuple[float, float, float]:
    """Convert legacy 2D position to 3D format."""
    if len(pos) == 2:
        return (pos[0], 0.0, pos[1])
    elif len(pos) == 3:
        return (pos[0], pos[1], pos[2])
    else:
        raise ValueError(f"Invalid position format: {pos}")


def _load_arrangement(glb_file_path: str, motif_name: str) -> Optional[Arrangement]:
    """Load arrangement from pickle file with fallback path resolution."""
    if not glb_file_path:
        return None

    cleaned_glb_path = clean_local_scratch_path(glb_file_path)
    pickle_paths_to_try = [Path(cleaned_glb_path).parent / f"{motif_name}.pkl"]

    # Try variations of the motif name
    import re
    base_name_match = re.match(r'^(.+)_(\d+)$', motif_name)
    if base_name_match:
        base_name = base_name_match.group(1)
        current_suffix = int(base_name_match.group(2))
        if current_suffix != 1:
            pickle_paths_to_try.append(Path(cleaned_glb_path).parent / f"{base_name}_1.pkl")
        pickle_paths_to_try.append(Path(cleaned_glb_path).parent / f"{base_name}.pkl")

    # Try each path until we find one that exists
    for pickle_path in pickle_paths_to_try:
        if pickle_path.exists():
            try:
                arrangement = Arrangement.load_pickle(str(pickle_path))
                print(f"Successfully loaded arrangement from: {pickle_path}")
                return arrangement
            except Exception as e:
                print(f"Failed to load arrangement from {pickle_path}: {e}")
                continue

    print(f"Warning: No arrangement pickle file found for motif {motif_name}")
    print(f"Tried paths: {[str(p) for p in pickle_paths_to_try]}")
    return None


def save_scene_state(
    room_desc: str,
    scene_motifs: List[SceneMotif],
    room_vertices: List[Tuple[float, float]],
    door_location: Tuple[float, float],
    room_type: str = "",
    filename: str = "scene_state.json",
    scene_spec: Optional[SceneSpec] = None,
    window_location: Optional[List[Tuple[float, float]]] = None,
    room_details: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    visualizations: Optional[Dict[str, str]] = None,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> None:
    """
    Save the current scene state to a JSON file

    Args:
        room_desc: Text description of the room
        scene_motifs: List of SceneMotif objects
        room_vertices: List of (x, y) coordinates defining room boundary
        door_location: (x, y) position of the door
        room_type: Type of room (e.g. bedroom, kitchen)
        filename: Path to save JSON file
        scene_spec: Optional SceneSpec containing furniture specifications
        window_location: Optional list of (x, y) positions of windows
        room_details: Optional string containing room details
        metrics: Optional performance metrics (time, memory, etc.)
        visualizations: Optional dict of visualization snapshot paths
        errors: Optional list of error messages
        warnings: Optional list of warning messages
    """
    project_root = Path(clean_local_scratch_path(str(Path(__file__).parent.parent.resolve())))
    unique_object_specs = {}
    unique_scene_objects = {}
    error_list = errors or []
    warning_list = warnings or []

    # Deduplicate object specs and scene objects
    for motif in scene_motifs:
        for spec in motif.object_specs:
            unique_object_specs[spec.id] = spec.to_dict()
        for obj in motif.objects:
            unique_scene_objects[obj.id] = obj.to_dict()

    motif_data_list = []
    output_dir = Path(filename).parent

    for motif in scene_motifs:
        motif_data = motif.to_dict()

        # Resolve GLB file path
        _resolve_motif_path(motif_data, "glb_file", project_root, output_dir, warning_list)

        # Handle arrangement pickle path
        if motif_data.get("glb_file"):
            glb_path = Path(motif_data["glb_file"])
            if not glb_path.is_absolute():
                glb_path = Path(project_root, motif_data["glb_file"])

            arrangement_pkl = glb_path.with_suffix(".pkl")
            if arrangement_pkl.exists():
                motif_data["arrangement_pickle"] = str(arrangement_pkl)
                _resolve_motif_path(motif_data, "arrangement_pickle", project_root, output_dir, warning_list)

        # Handle visualization snapshot
        if motif_data.get("glb_file"):
            glb_path = Path(motif_data["glb_file"])
            if not glb_path.is_absolute():
                glb_path = Path(project_root, motif_data["glb_file"])

            vis_path = glb_path.parent / "motif_top_down.png"
            if vis_path.exists():
                motif_data["visualization"] = str(vis_path)
                _resolve_motif_path(motif_data, "visualization", project_root, output_dir, warning_list)
        # Reference object specs and scene objects by ID
        motif_data["object_spec_ids"] = [spec.id for spec in motif.object_specs]
        motif_data["scene_object_ids"] = [obj.id for obj in motif.objects]
        motif_data.pop("object_specs", None)
        motif_data.pop("scene_objects", None)
        motif_data_list.append(motif_data)

    scene_state = {
        "scene_state_version": 1,
        "room_desc": room_desc,
        "room_type": room_type,
        "room_details": room_details,
        "room_vertices": numpy_to_python(room_vertices),
        "door_location": numpy_to_python(door_location),
        "window_location": numpy_to_python(window_location) if window_location else None,
        "scene_spec": scene_spec.to_dict() if scene_spec else None,
        "object_specs": unique_object_specs,
        "scene_objects": unique_scene_objects,
        "motifs": motif_data_list,
        "metrics": metrics or {},
        "visualizations": visualizations or {},
        "errors": error_list,
        "warnings": warning_list,
    }

    # Atomic write
    filename = str(filename)
    tmp_path = Path(filename).with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(scene_state, f, cls=CompactJSONEncoder, indent=4)
        Path(tmp_path).replace(filename)
        logger.info(f"Scene state saved to {filename}")
    except Exception as e:
        error_msg = f"Error saving scene state to {filename}: {e}"
        logger.error(error_msg)
        error_list.append(error_msg)
        # Fallback: save error info
        try:
            with open(filename, "w") as f:
                json.dump({"errors": error_list}, f, indent=4)
        except Exception as e2:
            logger.error(f"Failed to save error info: {e2}")

def legacy_load_scene_state(
    filename: str = "scene_state.json",
    object_types: Optional[list[ObjectType]] = None
) -> Tuple[str, List[SceneMotif], List[Tuple[float, float]], Tuple[float, float], str, Optional[SceneSpec], Optional[List[Tuple[float, float]]], str]:
    """
    Legacy loader for scene state files with 'objects' list (pre-deduplication format).
    Args:
        filename: Path to scene state JSON file
        object_types: list of ObjectType to filter motifs
    Returns:
        tuple containing:
            - room description (str)
            - list of SceneMotif objects
            - room vertices as (x,z) coordinates
            - door location as (x,z)
            - room type (str)
            - scene specification (optional)
            - window locations as list of (x,z) coordinates (optional)
            - room details (str)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Scene state file not found: {filename}")
    with open(filename, 'r') as f:
        scene_state = json.load(f)
    
    current_scene_dir = str(Path(filename).parent.resolve())
    current_dir = Path(clean_local_scratch_path(str(Path(filename).parent.resolve()), current_scene_dir))
    print(f"Current directory: {current_dir}\n")
    # breakpoint()
    scene_motifs = []
    for obj_data in scene_state["objects"]:
        scene_objects = []
        for obj in obj_data.get("scene_objects", []):
            try:
                pos = _convert_position_to_3d(obj["position"])
                scene_obj = SceneObject(
                    name=obj["name"],
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    dimensions=tuple(obj["dimensions"]),
                    rotation=obj["rotation"],
                    mesh_path=obj["mesh_path"],
                    obj_type=ObjectType(obj.get("obj_type", "undefined")),
                    id=obj.get("id")
                )
                scene_objects.append(scene_obj)
            except (ValueError, TypeError, OSError) as e:
                logger.warning(f"Failed to create SceneObject {obj.get('name', 'unknown')}: {e}")
                continue
        # Try to load arrangement if glb path exists
        arrangement = None
        if obj_data["glb_file"]:
            resolved_path = current_dir / obj_data['name'] / f"{obj_data['name']}.pkl"
            print(f"Arrangement path: {resolved_path}")
            if resolved_path.exists():
                try:
                    arrangement = Arrangement.load_pickle(str(resolved_path))
                except (OSError, ValueError, TypeError) as e:
                    raise ValueError(f"Failed to load arrangement: {e}")
            else:
                print(f"Arrangement file not found: {resolved_path}")
                arrangement_path = Path(obj_data["glb_file"]).parent / f"{obj_data['name']}.pkl"
                if arrangement_path.exists():
                    try:
                        arrangement = Arrangement.load_pickle(str(arrangement_path))
                    except (OSError, ValueError, TypeError) as e:
                        raise ValueError(f"Failed to load arrangement: {e}")
                else:
                    print(f"Warning: Arrangement file not found: {arrangement_path}")
                    arrangement = None

        try:
            pos = _convert_position_to_3d(obj_data["position"])
            object_specs = []
            for spec_dict in obj_data["object_specs"]:
                try:
                    object_specs.append(ObjectSpec(
                        id=spec_dict["id"],
                        name=spec_dict["name"],
                        description=spec_dict["description"],
                        dimensions=spec_dict["dimensions"],
                        amount=spec_dict["amount"],
                        parent_object=spec_dict.get("parent_object")
                    ))
                except (ValueError, TypeError, OSError) as e:
                    logger.warning(f"Failed to create ObjectSpec {spec_dict.get('name', 'unknown')}: {e}")
                    continue
            motif = SceneMotif(
                id=obj_data["name"],
                extents=tuple(obj_data["extents"]),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                rotation=obj_data["rotation"],
                description=obj_data["description"],
                glb_path=obj_data["glb_file"] if obj_data["glb_file"] else "",
                scene_objects=scene_objects,
                arrangement=arrangement,
                object_type=ObjectType(scene_objects[0].obj_type.value if scene_objects else "undefined"),
                object_specs=object_specs,
                ignore_collision=obj_data.get("ignore_collision", False)
            )
            if object_types and motif.object_type not in object_types:
                print(f"INFO: Skipping motif {motif.id} because it is not in the list of object types to load: {object_types}")
                continue
            scene_motifs.append(motif)
        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"Failed to create SceneMotif for {obj_data.get('name', 'unknown')}: {e}")
            continue
    scene_spec = None
    if "scene_spec" in scene_state and scene_state["scene_spec"]:
        scene_spec = SceneSpec.from_json(json.dumps(scene_state["scene_spec"]), required=True) # defaults as true in legacy format

    # Associate small object specs with their parent motifs
    if scene_spec and scene_spec.small_objects:
        # Create a map from large object spec IDs (within motifs) to their parent motifs
        large_spec_to_motif_map: Dict[int, SceneMotif] = {}
        for motif_item in scene_motifs:
            for large_spec in motif_item.object_specs: # These are ObjectSpec instances already in the motif
                if large_spec.id is not None:
                    large_spec_to_motif_map[large_spec.id] = motif_item

        for small_spec in scene_spec.small_objects: # These are ObjectSpec instances from hsm_core.scene.spec
            if small_spec.parent_object is not None:
                parent_motif = large_spec_to_motif_map.get(small_spec.parent_object)
                if parent_motif:
                    # Add the small object's spec to its parent motif's list of specs
                    if small_spec not in parent_motif.object_specs:
                        parent_motif.object_specs.append(small_spec)
                        logger.info(f"Associated small object spec '{small_spec.name}' (ID: {small_spec.id}) with motif '{parent_motif.id}'.")
                else:
                    logger.warning(
                        f"Could not find parent motif for small object spec '{small_spec.name}' (ID: {small_spec.id}) "
                        f"with parent_object ID: {small_spec.parent_object}. It will not be directly associated with a motif."
                    )
            else:
                logger.warning(
                    f"Small object spec '{small_spec.name}' (ID: {small_spec.id}) has no parent_object defined. "
                    "It will not be associated with a motif."
                )

    window_location = None
    if "window_location" in scene_state and scene_state["window_location"]:
        window_location = scene_state["window_location"]
    room_details = scene_state.get("room_details", "Room vertices: " + str(scene_state["room_vertices"]) + " Door location: " + str(scene_state["door_location"]) + " Window locations: " + str(window_location))
    return (
        scene_state["room_desc"],
        scene_motifs,
        scene_state["room_vertices"],
        tuple(scene_state["door_location"]),
        scene_state.get("room_type", ""),
        scene_spec,
        window_location,
        room_details
    )

def load_scene_state(
    filename: str = "scene_state.json",
    object_types: Optional[list[ObjectType]] = None
) -> Tuple[
    str, List[SceneMotif], List[Tuple[float, float]], Tuple[float, float], str, Optional[SceneSpec], Optional[List[Tuple[float, float]]], str, Optional[dict], Optional[list], Optional[list]
]:
    """
    Load scene state from a JSON file. Supports both new (deduplicated) and legacy formats.

    Args:
        filename: Path to scene state JSON file
        object_types: list of ObjectType to filter motifs
    Returns:
        tuple containing:
            - room description (str)
            - list of SceneMotif objects
            - room vertices as (x,z) coordinates
            - door location as (x,z)
            - room type (str)
            - scene specification (optional)
            - window locations as list of (x,z) coordinates (optional)
            - room details (str)
            - metrics (dict, optional)
            - errors (list, optional)
            - warnings (list, optional)
    """
    project_root = Path(clean_local_scratch_path(str(Path(__file__).parent.parent.resolve())))
    current_scene_dir = str(Path(filename).parent.resolve())

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Scene state file not found: {filename}")

    with open(filename, 'r') as f:
        scene_state = json.load(f)

    # Handle errors/warnings if present
    errors = scene_state.get("errors", [])
    warnings = scene_state.get("warnings", [])
    metrics = scene_state.get("metrics", {})

    # New format: deduplicated with 'motifs', 'object_specs', 'scene_objects'
    if "motifs" in scene_state and "object_specs" in scene_state and "scene_objects" in scene_state:
        # Reconstruct object specs and scene objects
        object_specs_dict = scene_state["object_specs"]
        scene_objects_dict = scene_state["scene_objects"]
        # Build objects by ID
        object_specs = {int(k): ObjectSpec(**v) for k, v in object_specs_dict.items()}
        scene_objects = {k: SceneObject(**v) for k, v in scene_objects_dict.items()}
        scene_motifs = []
        for motif_data in scene_state["motifs"]:
            # Resolve glb_file, arrangement_pickle, visualization to absolute
            for key in ("glb_file", "arrangement_pickle", "visualization"):
                if motif_data.get(key):
                    try:
                        # Clean up path by removing /local-scratch/ prefix if present
                        cleaned_path = clean_local_scratch_path(motif_data[key], current_scene_dir)
                        motif_data[key] = str(to_absolute_path(cleaned_path, current_scene_dir))
                    except Exception as e:
                        raise ValueError(f"Failed to resolve {key}: {motif_data.get(key)} ({e})")
            # Reconstruct object_specs and scene_objects for this motif
            motif_object_specs = [object_specs[oid] for oid in motif_data.get("object_spec_ids", []) if oid in object_specs]
            motif_scene_objects = [scene_objects[oid] for oid in motif_data.get("scene_object_ids", []) if oid in scene_objects]
            # Backward compatibility: if scene_objects list present, use it
            if motif_data.get("scene_objects"):
                motif_scene_objects = [SceneObject(**obj) for obj in motif_data["scene_objects"]]
            # Load arrangement if arrangement_pickle is present
            arrangement = None
            arrangement_pickle_path = motif_data.get("arrangement_pickle")
            if arrangement_pickle_path:
                try:
                    # Clean up path by removing /local-scratch/ prefix if present
                    cleaned_path = clean_local_scratch_path(arrangement_pickle_path, current_scene_dir)
                    arrangement_path = Path(cleaned_path)
                    if arrangement_path.exists():
                        arrangement = Arrangement.load_pickle(str(arrangement_path))
                    else:
                        print(f"Arrangement pickle does not exist: {cleaned_path}")
                        arrangement_path = to_absolute_path(cleaned_path, current_scene_dir)
                        if arrangement_path.exists():
                            arrangement = Arrangement.load_pickle(str(arrangement_path))
                        else:
                            raise ValueError(f"Arrangement pickle does not exist: {arrangement_path}")
                except Exception as e:
                    raise ValueError(f"Warning: Failed to load arrangement for motif {motif_data.get('name', 'unknown')}: {e}")
            else:
                print("Arrangement pickle not found, loading from GLB file")
                glb_file_path = motif_data.get("glb_file")
                motif_name = motif_data.get('name')
                arrangement = _load_arrangement(glb_file_path, motif_name)
                if arrangement is None:
                    raise ValueError(f"No arrangement pickle file found for motif {motif_name}")
            motif = SceneMotif(
                id=motif_data["name"],
                extents=tuple(motif_data["extents"]),
                position=tuple(motif_data["position"]),
                rotation=motif_data["rotation"],
                description=motif_data["description"],
                glb_path=motif_data.get("glb_file"),
                scene_objects=motif_scene_objects,
                arrangement=arrangement,
                object_type=ObjectType(motif_data.get("object_type", "undefined")),
                object_specs=motif_object_specs,
                ignore_collision=motif_data.get("ignore_collision", False)
            )
            if object_types and motif.object_type not in object_types:
                logger.info(f"Skipping motif {motif.id} because it is not in the list of object types to load: {object_types}")
                continue
            scene_motifs.append(motif)
            room_details = scene_state.get("room_details", "Room vertices: " + str(scene_state["room_vertices"]) + " Door location: " + str(scene_state["door_location"]) + " Window locations: " + str(scene_state.get("window_location")))
        return (
            scene_state["room_desc"],
            scene_motifs,
            scene_state["room_vertices"],
            tuple(scene_state["door_location"]),
            scene_state.get("room_type", ""),
            SceneSpec.from_json(json.dumps(scene_state["scene_spec"])) if scene_state.get("scene_spec") else None,
            scene_state.get("window_location", None),
            room_details,
            metrics,
            errors,
            warnings
        )
    # Legacy format: 'objects' list
    else:
        legacy_result = legacy_load_scene_state(filename, object_types)
        # Pad with None for metrics/errors/warnings for compatibility
        return (*legacy_result, None, None, None)

def load_glb_and_get_extents(file_path: Path) -> Tuple[float, float, float]:
    """
    Load a GLB file and return its extents.
    
    Args:
        file_path: Path to GLB file
        
    Returns:
        tuple: (x, y, z) extents
    """
    try:
        if not file_path.exists():
            print(f"File does not exist: {file_path}")
            return (0, 0, 0)
            
        scene = trimesh.load(str(file_path), force='mesh')
        if isinstance(scene, trimesh.Scene):
            # Get all meshes from the scene
            meshes = []
            for geometry in scene.geometry.values():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            
            if not meshes:
                print(f"No valid meshes found in {file_path}")
                return (0, 0, 0)
                
            # Combine all meshes
            combined_mesh = trimesh.util.concatenate(meshes)
            if hasattr(combined_mesh, 'bounding_box'):
                extents = combined_mesh.bounding_box.extents
            else:
                # Fallback: calculate extents manually
                all_vertices = np.concatenate([mesh.vertices for mesh in meshes])
                min_coords = np.min(all_vertices, axis=0)
                max_coords = np.max(all_vertices, axis=0)
                extents = max_coords - min_coords
            print(f"Loaded {file_path} - Extents: {extents}")
            return tuple(extents)
        elif isinstance(scene, trimesh.Trimesh):
            extents = scene.bounding_box.extents
            print(f"Loaded {file_path} - Extents: {extents}")
            return tuple(extents)
        else:
            print(f"Unexpected scene type for {file_path}: {type(scene)}")
            return (0, 0, 0)
    except (OSError, ValueError, TypeError) as e:
        print(f"Error loading {file_path}: {str(e)}")
        return (0, 0, 0)


def room_to_world(pos: Tuple[float, float, float], scene_offset: List[float]) -> np.ndarray:
    """
    Transform a point from room coordinates to world coordinates using a unified transformation.

    Args:
        pos: (x, y, z) in room space.
             - In Room Space, the origin is at the room's bottom-left corner;
               x increases rightward and z increases forward.
        scene_offset: Offset computed from room bounds

    Returns:
        np.ndarray: (x, y, z) in World Space, where:
            - X is right,
            - Y is upward (with a small constant added so objects sit on the floor),
            - Z is computed by negating the room space z and adding the corresponding offset.
    """
    return np.array([
        float(pos[0]) + scene_offset[0],
        float(pos[1]) + scene_offset[1],
        float(pos[2]) + scene_offset[2]
    ])


def create_front_arrow(front_vector: np.ndarray, length: float = 0.5, thickness: float = 0.05) -> trimesh.Trimesh:
    """
    Create an arrow mesh pointing in the given front_vector.

    Args:
        front_vector: 3D numpy array indicating the arrow's direction.
        length: Total length of the arrow.
        thickness: Overall thickness (diameter) of the arrow shaft.

    Returns:
        trimesh.Trimesh: Combined mesh for the arrow.
    """
    # Define parts: shaft (cylinder) and arrowhead (cone)
    shaft_length: float = length * 0.7  # ARROW_SHAFT_PROPORTION
    cone_length: float = length * 0.3  # ARROW_HEAD_PROPORTION
    shaft_radius: float = thickness * 0.5
    cone_radius: float = thickness

    # Create shaft: a cylinder originally centered at the origin.
    shaft: trimesh.Trimesh = trimesh.creation.cylinder(
        radius=shaft_radius,
        height=shaft_length,
        sections=32
    )
    # Translate shaft so that its base is at origin and it extends in +Z.
    shaft.apply_translation([0, 0, shaft_length / 2])

    # Create arrowhead as a cone.
    cone: trimesh.Trimesh = trimesh.creation.cone(
        radius=cone_radius,
        height=cone_length,
        sections=32
    )
    # Translate cone so its base touches the top of the shaft.
    cone.apply_translation([0, 0, shaft_length + (cone_length / 2)])

    # Combine shaft and cone.
    arrow: trimesh.Trimesh = trimesh.util.concatenate([shaft, cone])

    # The arrow is by default oriented along +Z.
    # Compute rotation matrix to align +Z to the provided front_vector.
    default_dir: np.ndarray = np.array([0, 0, 1])
    target_dir: np.ndarray = front_vector / np.linalg.norm(front_vector)
    axis: np.ndarray = np.cross(default_dir, target_dir)
    if np.linalg.norm(axis) < 1e-6:
        # Vectors are parallel or anti-parallel.
        if np.dot(default_dir, target_dir) < 0:
            rot: np.ndarray = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        else:
            rot = np.eye(4)
    else:
        rot_angle: float = np.arccos(np.clip(np.dot(default_dir, target_dir), -1, 1))
        rot: np.ndarray = trimesh.transformations.rotation_matrix(rot_angle, axis)
    arrow.apply_transform(rot)

    # Optionally, color the arrow (red, with full opacity)
    if hasattr(arrow, 'visual'):
        arrow.visual.face_colors = [255, 0, 0, 255]

    return arrow
