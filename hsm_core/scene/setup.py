"""
Scene setup and initialization utilities.

This module handles the initialization of scenes from configuration,
loading existing scenes, and setting up the necessary components.
"""

import json
from pathlib import Path
from typing import Any, Optional, Tuple, List, Sequence
from dataclasses import dataclass

from omegaconf import DictConfig
from shapely.geometry import Polygon

from hsm_core.vlm.gpt import extract_json
from hsm_core.vlm.vlm import create_session
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hsm_core.vlm.gpt import Session
from hsm_core.utils.io import create_output_directory, get_next_iteration, setup_logging
from hsm_core.scene.manager import Scene
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.spec import SceneSpec
from hsm_core.scene.validate import validate_floorplan, validate_room_type
from hsm_core.scene.visualization import SceneVisualizer
from hsm_core.config import PROMPT_DIR, PROJECT_ROOT

def _get_model_type_from_config(cfg: Optional[DictConfig]) -> str:
    """Get the VLM model type from configuration, defaulting to 'gpt'."""
    if cfg is None:
        return 'gpt'
    llm_config = getattr(cfg, 'llm', None)
    if llm_config is None:
        return 'gpt'
    return getattr(llm_config, 'model_type', 'gpt')


def _get_enable_spatial_optimization(cfg: DictConfig) -> bool:
    """Get the enable_spatial_optimization setting from configuration, defaulting to True."""
    return getattr(cfg.mode, 'enable_spatial_optimization', True)


def _normalize_room_type(room_type: str) -> str:
    """Normalize room type string for filesystem compatibility."""
    return room_type.replace(" ", "_").replace(",", "").replace(".", "").replace("'", "_")

@dataclass
class SceneSetupResult:
    """Container for scene setup results."""
    scene: Scene
    output_dir: Path
    room_session: Optional["Session"] = None
    visualizer: Optional[SceneVisualizer] = None
    sessions_dir: Optional[Path] = None
    is_loaded_scene: bool = False
    logger: Any = None


@dataclass(frozen=True)
class RoomGeometry:
    """Container for resolved room geometry data."""
    vertices: List[Tuple[float, float]]
    height: float
    door_location: Optional[Tuple[float, float]]
    window_locations: Optional[List[Tuple[float, float]]]


def _as_xy_list(val: Optional[Sequence[Sequence[float]]]) -> Optional[List[Tuple[float, float]]]:
    """Convert sequence of 2D coordinates to list of tuples."""
    if not val:
        return None
    return [tuple(map(float, xy)) for xy in val]


def _as_xy(val: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    """Convert 2D coordinate sequence to tuple."""
    if not val:
        return None
    return tuple(map(float, val))  # type: ignore[return-value]


def resolve_room_geometry(
    cfg: DictConfig,
    room_session: "Session",
    room_description: str,
    room_type: str,
) -> RoomGeometry:
    """
    Resolve room geometry from config or generate via VLM.

    Args:
        cfg: Configuration object
        room_session: Session for VLM calls
        room_description: Description of the room
        room_type: Type of the room
        generate_when: When to generate - "all_missing" or "any_missing"

    Returns:
        RoomGeometry with resolved values
    """
    vertices = _as_xy_list(getattr(cfg.room, "vertices", None))
    door = _as_xy(getattr(cfg.room, "door_location", None))
    windows = _as_xy_list(getattr(cfg.room, "window_locations", None))
    height = getattr(cfg.room, "height", None)
    if height is None:
        height = 2.5

    need_generation = vertices is None or door is None or windows is None

    if need_generation:
        payload = {
            "room_description": room_description,
            "room_type": room_type,
            "room_vertices": vertices,
            "door_location": door,
            "window_locations": windows,
            "room_height": height,
        }
        resp = room_session.send_with_validation(
            "room_boundary",
            payload,
            validate_floorplan,
            verbose=True,
            json=True,
        )
        data = json.loads(extract_json(resp))
        # Preserve config values, only use VLM for missing fields
        vertices = vertices or _as_xy_list(data.get("room_vertices"))
        height = height if height is not None else (float(data["room_height"]) if "room_height" in data else None)
        door = door or _as_xy(data.get("door_location"))
        windows = windows or _as_xy_list(data.get("window_locations"))

    return RoomGeometry(
        vertices=vertices or [],
        height=float(height) if height is not None else 0.0,
        door_location=door,
        window_locations=windows,
    )


def determine_scene_paths(cfg: DictConfig) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Determine paths for loading existing scenes based on configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (path_to_load_state_from, base_folder_for_output_naming)
    """
    path_to_load_state_from: Optional[Path] = None
    base_folder_for_output_naming: Optional[Path] = None
    
    if getattr(cfg.execution, 'load_specific_folder', None):
        specific_folder = Path(cfg.execution.load_specific_folder)
        print(f"Loading scene from specified folder: {specific_folder}")
        if not specific_folder.exists():
            raise ValueError(f"Specified load folder not found: {specific_folder}")

        path_to_load_state_from = specific_folder
        base_folder_for_output_naming = specific_folder

    elif getattr(cfg.execution, 'use_previous_result', False):
        print(f"Using previous scene from: {cfg.execution.result_dir}")
        result_dir = Path(PROJECT_ROOT / cfg.execution.result_dir) if cfg.execution.result_dir else Path(PROJECT_ROOT / "scene" / "result")
        if not result_dir.exists():
            raise ValueError(f"Result directory not found: {result_dir}")

        potential_folders = sorted(
            [
                d
                for d in result_dir.iterdir()
                if d.is_dir() and not d.name.startswith("_") and not "iteration" in d.name
            ],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not potential_folders:
            raise ValueError(f"No previous results found in {result_dir}")

        base_folder_for_output_naming = potential_folders[0]
        path_to_load_state_from = base_folder_for_output_naming
        print(f"Using latest scene found: {base_folder_for_output_naming}")
    
    return path_to_load_state_from, base_folder_for_output_naming


def load_existing_scene(
    cfg: DictConfig,
    path_to_load_state_from: Path,
    base_folder_for_output_naming: Path
) -> Tuple[Scene, Path]:
    """
    Load an existing scene from state file.
    
    Args:
        cfg: Configuration object
        path_to_load_state_from: Path to load scene state from
        base_folder_for_output_naming: Base folder for output naming
        
    Returns:
        Tuple of (scene, output_dir)
    """
    next_iteration = get_next_iteration(base_folder_for_output_naming)
    print(f"Starting iteration {next_iteration} based on {base_folder_for_output_naming.name}\n")

    output_dir = base_folder_for_output_naming.parent / f"{base_folder_for_output_naming.name}_iteration_{next_iteration}"
    
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    
    full_scene_state_path = path_to_load_state_from / "scene_state_full.json"
    scene_state_path = path_to_load_state_from / "scene_state.json"
    if not full_scene_state_path.exists():
        print(f"Full scene state file not found, using scene state file: {scene_state_path}")
        full_scene_state_path = scene_state_path

    if not full_scene_state_path.exists():
        raise ValueError(f"Scene state file not found in: {full_scene_state_path}")
    
    load_object_types: List[ObjectType] = []
    if getattr(cfg.mode, 'load_object_types', None) is not None:
        for object_type in cfg.mode.load_object_types:
            load_object_types.append(ObjectType(object_type))

    print(f"Loading scene state from: {full_scene_state_path}")
    scene = Scene.from_scene_state(full_scene_state_path, object_types=load_object_types)
    if not scene:
        raise ValueError("Failed to load scene from state file.")

    scene.enable_spatial_optimization = _get_enable_spatial_optimization(cfg)
    
    if not scene.room_polygon:
        if scene.room_vertices:
            Polygon(scene.room_vertices)  # Validation check
        else:
            raise ValueError("Room geometry (polygon/vertices) missing from loaded scene state.")
    
    return scene, output_dir


def create_new_scene_from_config(
    cfg: DictConfig,
    project_root: Path,
    output_dir_name_override: Optional[str] = None,
    timestamp: bool = True
) -> Tuple[Scene, Path, "Session"]:
    """
    Create a new scene from configuration.
    
    Args:
        cfg: Configuration object
        project_root: Project root path
        output_dir_name_override: Override for output directory name
        timestamp: Whether to include timestamp in output directory
        
    Returns:
        Tuple of (scene, output_dir, room_session)
    """
    room_description = cfg.room.room_description
    model_type = _get_model_type_from_config(cfg)
    room_session = create_session(str(PROMPT_DIR / "scene_prompts_room.yaml"), model_type=model_type)
    
    # Determine room type
    room_type_response = room_session.send_with_validation(
        "room_type", {"room_description": room_description}, validate_room_type, verbose=True, json=True
    )
    parsed_room_type_json = json.loads(extract_json(room_type_response))
    room_type = _normalize_room_type(parsed_room_type_json["room_type"])

    # Get room geometry from config or generate it
    geom = resolve_room_geometry(cfg, room_session, room_description, room_type)
    room_vertices = geom.vertices
    room_height = geom.height
    door_location = geom.door_location
    window_locations = geom.window_locations

    result_dir = cfg.execution.get('result_dir', getattr(cfg.execution, 'result_dir', 'results'))
    if output_dir_name_override:
        output_dir = create_output_directory(base_dir=result_dir, subfix=output_dir_name_override, timestamp=timestamp, project_root=project_root)
    else:
        output_dir = create_output_directory(base_dir=result_dir, subfix=room_type, timestamp=timestamp, project_root=project_root)

    scene = Scene(
        room_vertices,
        door_location,
        room_height=room_height,
        room_description=room_description,
        room_type=room_type,
        window_location=window_locations,
        enable_spatial_optimization=_get_enable_spatial_optimization(cfg),
    )
    
    return scene, output_dir, room_session


def setup_scene_environment(
    scene: Scene,
    output_dir: Path,
    project_root: Path,
    room_session: Optional["Session"] = None,
    cfg: Optional[DictConfig] = None
) -> Tuple[SceneVisualizer, "Session", Path, object]:
    """
    Set up the scene environment including logging, sessions, and visualization.
    
    Args:
        scene: Scene object
        output_dir: Output directory path
        project_root: Project root path
        room_session: Optional existing room session
        
    Returns:
        Tuple of (visualizer, room_session, sessions_dir, logger)
    """
    logger, _ = setup_logging(output_dir)
    sessions_dir = output_dir / ".sessions"
    sessions_dir.mkdir(exist_ok=True)
    
    model_type = _get_model_type_from_config(cfg) if cfg else 'gpt'
    from hsm_core.vlm.gpt import Session
    Session.set_global_output_dir(str(sessions_dir))

    if room_session is None:
        room_session = create_session(str(PROMPT_DIR / "scene_prompts_room.yaml"), model_type=model_type)
    room_session.output_dir = str(sessions_dir)

    visualizer = SceneVisualizer(scene)
    
    return visualizer, room_session, sessions_dir, logger


def initialize_scene_from_config(
    cfg: DictConfig,
    project_root: Path,
    output_dir_name_override: Optional[str] = None,
    output_dir_override: Optional[Path] = None,
    timestamp: bool = True
) -> SceneSetupResult:
    """
    Initialize a scene from configuration, handling both new and existing scenes.
    
    Args:
        cfg: Configuration object
        project_root: Project root path
        output_dir_name_override: Override for output directory name
        output_dir_override: Override for complete output directory path
        timestamp: Whether to include timestamp in output directory
        
    Returns:
        SceneSetupResult containing all setup components
    """
    path_to_load_state_from, base_folder_for_output_naming = determine_scene_paths(cfg)
    
    if path_to_load_state_from and base_folder_for_output_naming:
        scene, output_dir = load_existing_scene(cfg, path_to_load_state_from, base_folder_for_output_naming)

        if output_dir_override:
            output_dir_override.mkdir(parents=True, exist_ok=True)
            output_dir = output_dir_override
        
        room_session = None
        is_loaded_scene = True
    else:
        scene, output_dir, room_session = create_new_scene_from_config(
            cfg, project_root, output_dir_name_override, timestamp
        )
        if output_dir_override:
            output_dir_override.mkdir(parents=True, exist_ok=True)
            output_dir = output_dir_override
        is_loaded_scene = False

    visualizer, room_session, sessions_dir, logger = setup_scene_environment(
        scene, output_dir, project_root, room_session, cfg
    )
    
    return SceneSetupResult(
        scene=scene,
        output_dir=output_dir,
        room_session=room_session,
        visualizer=visualizer,
        sessions_dir=sessions_dir,
        is_loaded_scene=is_loaded_scene,
        logger=logger
    )


def perform_room_analysis_and_decomposition(
    scene: Scene,
    room_session: "Session",
    project_root: Path,
    visualizer: SceneVisualizer,
    output_dir: Path,
    cfg: Optional[DictConfig] = None
) -> Tuple[Any, str]:
    """
    Perform room analysis and scene decomposition for new scenes.
    
    Args:
        scene: Scene object
        room_session: Room session for VLM calls
        project_root: Project root path
        visualizer: Scene visualizer
        output_dir: Output directory
        
    Returns:
        Tuple of (initial_plot_path, room_details)
    """
    initial_plot, _ = visualizer.visualize(
        output_path=str(output_dir / "initial_plot.png"), add_grid_markers=True
    )

    room_details = room_session.send(
        "describe_room",
        {
            "room_vertices": scene.room_vertices,
            "door_location": scene.door_location,
            "window_locations": scene.window_location if scene.window_location else "No windows",
            "room_type": scene.room_type
        },
        verbose=True,
        images=initial_plot,
    )
    scene.room_details = room_details

    model_type = _get_model_type_from_config(cfg) if cfg else 'gpt'
    decompose_session = create_session(str(PROMPT_DIR / "scene_prompts_large.yaml"), model_type=model_type)
    objects_response = decompose_session.send(
        "requirements_decompose", {"room_description": scene.room_description}, json=True, verbose=True
    )
    scene.scene_spec = SceneSpec.from_json(objects_response, required=True)
    
    return initial_plot, room_details
