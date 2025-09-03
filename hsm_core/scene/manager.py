from __future__ import annotations
import os
from pathlib import Path
import traceback
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Union, Optional, Dict, Any
import json
import trimesh
import matplotlib.pyplot as plt

from hsm_core.config import PROMPT_DIR
from hsm_core.vlm.gpt import Session, extract_json
from hsm_core.scene.room_geometry import create_custom_room
from hsm_core.scene.validate import *
from hsm_core.scene.utils import load_scene_state, save_scene_state
from hsm_core.scene.motif import SceneMotif, filter_motifs_by_types
from hsm_core.scene.anchor import find_anchor_object
from hsm_core.scene.small_obj import (
    collect_surface_data, 
    optimize_small_objects, 
    populate_furniture, 
    update_small_motifs_from_constrained_layout,
    clean_layer_info
)
from hsm_core.scene.scene_3d import SceneObjectPlacer, create_scene_from_motifs, Cutout
from hsm_core.support_region.loader import check_support_json_exists
from hsm_core.scene.objects import SceneObject
from hsm_core.scene.generate_scene_motif import process_scene_motifs
from hsm_core.utils.stk_utils import save_stk_scene_state
from hsm_core.scene.spec import ObjectSpec, SceneSpec
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.ablation import create_individual_scene_motifs_with_analysis
from hsm_core.constants import *

def _get_all_motifs_recursive(motifs: list[SceneMotif]) -> list[SceneMotif]:
    """Recursively collect all motifs from a list of motifs and their children."""
    all_motifs = []
    for motif in motifs:
        all_motifs.append(motif)
        # Check for child motifs in objects
        for obj in motif.objects:
            if hasattr(obj, 'child_motifs') and obj.child_motifs:
                all_motifs.extend(_get_all_motifs_recursive(obj.child_motifs))
    return all_motifs

class Scene:
    """
    3D scene manager that handles room geometry, motifs, and scene generation.
    """
    def __init__(
        self,
        room_vertices: Optional[list[tuple[float, float]]] = None,
        door_location: tuple[float, float] = (0, 1),
        window_location: Optional[list[tuple[float, float]]] = None,
        room_height: float = WALL_HEIGHT,
        room_details: str = "",
        scene_motifs: Optional[list[SceneMotif]] = None,
        room_description: str = "",
        room_type: str = "",
        scene_spec: Optional[SceneSpec] = None,
        enable_spatial_optimization: bool = True,
    ):
        """
        Initialize a Scene object.

        Args:
            room_vertices: Optional list of (x, y) tuples defining the room polygon. Defaults to a 3x3 square.
            door_location: (x, y) tuple for the door location. Defaults to (0, 1).
            window_location: Optional list of (x, y) tuples for window locations. Defaults to [(2, 2)] if not provided.
            room_height: Height of the room in meters. Defaults to 2.5.
            room_details: Additional details about the room (e.g., style, notes).
            scene_motifs: Optional list of SceneMotif objects to add to the scene.
            room_description: Textual description of the room.
            room_type: Type/category of the room (e.g., "bedroom", "kitchen").
            scene_spec: Optional SceneSpec object with detailed scene specification.
        """
        self.room_vertices = room_vertices or [(0, 0), (0, 3), (3, 3), (3, 0)]
        self.room_polygon = create_custom_room(self.room_vertices)
        self.room_height = room_height
        self.room_description = room_description
        self.room_type = room_type
        self.room_details = room_details
        
        self.room_plot = None
        self.door_location = self._validate_door_location(door_location)
        self.window_location = self._validate_window_location(window_location)

        self.scene: Optional[trimesh.Scene] = None
        self.scene_placer: Optional[SceneObjectPlacer] = None
        self.scene_spec: Optional[SceneSpec] = scene_spec
        self.enable_spatial_optimization = enable_spatial_optimization

        self._scene_motifs: list[SceneMotif] = []
        if scene_motifs:
            self.add_motifs(scene_motifs)
    
    def add_motifs(self, motifs: list[SceneMotif]) -> None:
        existing_ids: set = {motif.id for motif in self._scene_motifs}
        for m in motifs:
            if m.id in existing_ids:
                print(f"Warning: Motif {m.id} already exists in scene")
                continue
            self._scene_motifs.append(m)
            existing_ids.add(m.id)
        # Invalidate scene cache since motifs have been modified
        self._invalidate_scene_cache()

    def _invalidate_scene_cache(self) -> None:
        """Invalidate the cached scene and scene_placer when motifs are modified."""
        self.scene = None
        self.scene_placer = None
        # Also clear cached normalized data that depends on scene_placer
        if hasattr(self, '_cached_normalized_cutouts'):
            delattr(self, '_cached_normalized_cutouts')
        # Clear spatial optimizer cache when scene structure changes
        self._clear_spatial_optimizer_cache()
    
        
    def _clear_spatial_optimizer_cache(self) -> None:
        """Clear the cached spatial optimizer to force recreation with updated scene."""
        if hasattr(self, '_cached_spatial_optimizer'):
            self._cached_spatial_optimizer = None
            print("  Cleared cached spatial optimizer due to scene changes")
    
    def invalidate_scene(self) -> None:
        """
        Public method to invalidate the scene cache.
        Call this when you've modified motifs externally and want to force scene regeneration.
        """
        self._invalidate_scene_cache()
    
    def is_scene_created(self) -> bool:
        """
        Check if the scene has been created and is available.
        
        Returns:
            bool: True if scene exists and is ready for use, False otherwise
        """
        return self.scene is not None and self.scene_placer is not None
    
        
    @classmethod
    def from_scene_state(
        cls,
        scene_state_path: Union[str, Path],
        object_types: Optional[list[ObjectType]] = None
    ) -> "Scene":
        """
        Create Scene instance from a scene state file.
        
        Args:
            scene_state_path: Path to the scene state JSON file
            object_types: Optional list of ObjectType to filter which object types to load
        
        Returns:
            Scene: New Scene instance with loaded state
        
        Raises:
            FileNotFoundError: If scene state file doesn't exist
            ValueError: If scene state file is invalid
        """
        scene_state_path = Path(scene_state_path)
        if not scene_state_path.exists():
            raise FileNotFoundError(f"Could not find scene state file: {scene_state_path}")

        room_desc, scene_motifs, vertices, door, room_type, scene_spec, window_location, room_details, metrics, errors, warnings = load_scene_state(str(scene_state_path), object_types)

        return cls(
            room_vertices=vertices,
            door_location=door,
            window_location=window_location,
            scene_motifs=scene_motifs,
            room_description=room_desc,
            room_type=room_type,
            scene_spec=scene_spec,
            room_details=room_details,
            enable_spatial_optimization=True  # Default to enabled for loaded scenes
        )
        
    @property
    def scene_motifs(self) -> list[SceneMotif]:
        """
        Get the list of scene motifs.

        Returns:
            list[SceneMotif]: The motifs currently in the scene.
        """
        return self._scene_motifs
        
    @scene_motifs.setter
    def scene_motifs(self, motifs: list[SceneMotif]):
        self._scene_motifs = []
        if motifs:
            for motif in motifs:
                self._scene_motifs.append(motif)
        # Invalidate scene cache since motifs have been replaced
        self._invalidate_scene_cache()
        # Note: No need to invalidate existing motifs since we're replacing all of them
    
    def save(self, output_dir: Path, suffix: str = "", recreate_scene: bool = False, save_scene_state: bool = True):
        """
        Save the scene to output directory.
        
        Args:
            output_dir: Directory to save to
            suffix: Suffix for filename
            recreate_scene: Whether to recreate the scene before saving
            save_scene_state: Whether to save scene state (controls multiple saves during pipeline)
        """
        if recreate_scene:
            self.invalidate_scene()
        
        if not self.is_scene_created():
            self.create_scene()
        
        # Export 3D scene
        if suffix:
            glb_filename = f"room_scene_{suffix}.glb"
            scene_state_filename = f"hsm_scene_state_{suffix}.json"
            stk_state_filename = f"stk_scene_state_{suffix}.json"
        else:
            glb_filename = "room_scene.glb"
            scene_state_filename = "hsm_scene_state.json"
            stk_state_filename = "stk_scene_state.json"
        
        self.export(output_dir / glb_filename, recreate_scene=False)
        
        # Save scene state only if requested (to avoid multiple saves during pipeline)
        # if save_scene_state:
            # self.save_state(output_dir / scene_state_filename)
            
        # Always save STK state for compatibility
        self.save_stk_state(output_dir / stk_state_filename)
    
    def create_scene(self) -> None:
        """
        Create a trimesh scene from the current state.
        
        This method generates the 3D scene and scene_placer from the current motifs.
        The scene is cached and will persist until motifs are modified or invalidate_scene() is called.
        
        Note: This method is automatically called by export(), save(), and populate_small_objects()
        when needed. Manual calls are typically not necessary unless you want to force regeneration.
        """
        
        # self._scene_motifs = [m for m in self._scene_motifs if m.object_type != ObjectType.UNDEFINED]
        top_level_motifs = self.scene_motifs
        all_objects = self.get_all_objects()
        
        print(f"Creating 3D scene with {len(all_objects)} objects from {len(top_level_motifs)} top-level motifs\n")
        
        door_loc = self.door_location if self.door_location is not None else (0, 0)
        
        # Create scene from motifs (spatial optimization is handled during stage processing)
        print("Creating scene from motifs...")
        enable_spatial_optimization = getattr(self, 'enable_spatial_optimization', True)
        self.scene, self.scene_placer = create_scene_from_motifs(
            top_level_motifs, 
            self.room_polygon, 
            door_loc, 
            self.window_location, 
            self.room_height,
            enable_spatial_optimization=enable_spatial_optimization,
            scene_manager=self  # Pass self for scene properties access
        )
        
        if self.scene and self.scene_placer:
            print(f"Final scene contains {len(self.scene_placer.placed_objects)} objects")
    

        
    def export(self, output_path: Path, recreate_scene: bool = False) -> None:
        """
        Export the scene to a GLB file.

        Args:
            output_path (Path): The path to export the GLB file.
            recreate_scene (bool): If True, reruns layout and recreates the scene even if self.scene exists.
        """
        if self.scene is None or recreate_scene:
            self.create_scene()
        print("Exporting GLB to", output_path)
        if self.scene:
            try:
                self.scene.export(output_path)
            except (OSError, AttributeError) as e:
                print(f"Error exporting scene: {e}")
                traceback.print_exc()
        else:
            print("Warning: Scene not created, cannot export GLB.")
            
    def save_state(self, output_path: Path) -> None:
        """Save the current scene state to a JSON file."""
        all_motifs = self.get_all_motifs()
        
        try:
            save_scene_state(
                room_desc=self.room_description,
                scene_motifs=self._scene_motifs,
                room_vertices=self.room_vertices,
                door_location=self.door_location,
                room_type=self.room_type,
                filename=str(output_path),
                scene_spec=self.scene_spec,
                window_location=self.window_location or None,
                room_details=self.room_details
            )
            print(f"Scene state saved to {output_path} with {len(all_motifs)} motifs")
        except Exception as e:
            print(f"Error saving scene state: {e}")
            traceback.print_exc()
    
    def _normalize_vertices(self) -> tuple[list[list[float]], float, float]:
        if hasattr(self, '_cached_normalized_vertices'):
            return self._cached_normalized_vertices
        from shapely.geometry import Polygon
        polygon = Polygon(self.room_vertices)
        minx = polygon.bounds[0]
        miny = polygon.bounds[1]
        normalized_room_vertices = [[v[0] - minx, v[1] - miny] for v in self.room_vertices]
        self._cached_normalized_vertices = (normalized_room_vertices, minx, miny)
        return self._cached_normalized_vertices

    def _get_normalized_cutouts(self, minx: float, miny: float) -> tuple[object, object]:
        if hasattr(self, '_cached_normalized_cutouts'):
            return self._cached_normalized_cutouts
        door_cutout = getattr(self.scene_placer, 'door_cutout', None)
        window_cutouts = getattr(self.scene_placer, 'window_cutouts', None)
        from hsm_core.scene.scene_3d import Cutout
        # Door
        if door_cutout:
            normalized_door_cutout = Cutout(
                location=(door_cutout.location[0] - minx, door_cutout.location[1] - miny),
                cutout_type=door_cutout.cutout_type,
                width=door_cutout.width,
                height=door_cutout.height
            )
            normalized_door_cutout.closest_wall_index = door_cutout.closest_wall_index
            normalized_door_cutout.projection_on_wall = door_cutout.projection_on_wall
            door_location = normalized_door_cutout
        else:
            normalized_door_location = (
                (self.door_location[0] - minx, self.door_location[1] - miny)
                if self.door_location is not None else None
            )
            door_location = normalized_door_location
        # Windows
        if window_cutouts:
            normalized_window_cutouts = []
            for cutout in window_cutouts:
                normalized_cutout = Cutout(
                    location=(cutout.location[0] - minx, cutout.location[1] - miny),
                    cutout_type=cutout.cutout_type,
                    width=cutout.width,
                    height=cutout.height,
                    bottom_height=cutout.bottom_height
                )
                normalized_cutout.closest_wall_index = cutout.closest_wall_index
                normalized_cutout.projection_on_wall = cutout.projection_on_wall
                normalized_window_cutouts.append(normalized_cutout)
            window_locations = normalized_window_cutouts
        elif hasattr(self, 'window_location') and self.window_location:
            normalized_window_locations = [(w[0] - minx, w[1] - miny) for w in self.window_location]
            window_locations = normalized_window_locations
        else:
            window_locations = None
        self._cached_normalized_cutouts = (door_location, window_locations)
        return self._cached_normalized_cutouts

    def save_stk_state(self, output_path: Path) -> None:
        """Save the scene state in STK format."""
        # Ensure scene is created before accessing scene_placer
        if self.scene_placer is None:
            if self.scene is None:
                self.create_scene()

        if self.scene_placer:
            stk_objects = [
                (obj["id"], list(obj["position"]), obj["rotation"], obj["transform_matrix"])
                for obj in self.scene_placer.placed_objects
                if all(k in obj for k in ["id", "position", "rotation", "transform_matrix"])
            ]
        else:
            print("Warning: Could not create scene_placer, saving STK state with empty objects list")
            stk_objects = []
        
        try:
            normalized_room_vertices, minx, miny = self._normalize_vertices()
            door_location, window_locations = self._get_normalized_cutouts(minx, miny)
        except (ValueError, AttributeError) as e:
            print(f"Error normalizing room vertices or cutouts: {e}")
            normalized_room_vertices = self.room_vertices
            door_location = self.door_location
            window_locations = self.window_location if hasattr(self, 'window_location') else None
        
        # Extract filename from the path and pass directory separately
        filename = output_path.name
        output_dir = output_path.parent
        
        save_stk_scene_state(stk_objects, normalized_room_vertices, door_location, output_dir,
                             window_locations=window_locations, filename=filename)
    
    def _setup_small_objects_population(self) -> bool:
        """Setup and validate conditions for small object population."""
        if self.scene is None:
            self.create_scene()

        if not self.scene_spec:
            print("Warning: SceneSpec is not initialized. Cannot populate small objects.")
            return False
        return True

    def _get_constrained_small_objects(self) -> tuple[list, set]:
        """Get constrained small objects and their parent motifs."""
        constrained_small_objects = [
            obj for obj in self.scene_spec.small_objects
            if obj.parent_object is not None and obj.required
        ]
        constrained_parent_motifs = {obj.parent_object for obj in constrained_small_objects}

        print("########################################################")
        print("Processing Constrained Small Objects")
        print(f"Found {len(constrained_small_objects)} constrained small objects", constrained_small_objects, "\n")

        return constrained_small_objects, constrained_parent_motifs

    def _build_parent_name_to_id_map(self, motif: SceneMotif, constrained_parent_motifs: set = None) -> dict[str, int]:
        """Build mapping from instance names to object IDs for a motif."""
        parent_name_to_id_map = {}
        name_counts = {}

        for spec_in_motif in motif.object_specs:
            # Skip if we have constraints and this object is not a constrained parent
            if constrained_parent_motifs is not None and spec_in_motif.id not in constrained_parent_motifs:
                continue

            base_name = spec_in_motif.name
            if base_name in name_counts:
                name_counts[base_name] += 1
                instance_name = f"{base_name}_{name_counts[base_name]}"
            else:
                name_counts[base_name] = 1
                instance_name = base_name  # First instance keeps the original name
            parent_name_to_id_map[instance_name] = spec_in_motif.id

        return parent_name_to_id_map

    def _group_instances_by_base_name(self, parent_name_to_id_map: dict[str, int]) -> dict[str, list[str]]:
        """Group instance names by their base name using batch object spec lookups."""
        base_name_to_instance_names = {}

        for instance_name, obj_id in parent_name_to_id_map.items():
            parent_obj_spec = self.scene_spec.get_object_by_id(obj_id)
            if not parent_obj_spec:
                continue
            base_name = parent_obj_spec.name
            if base_name not in base_name_to_instance_names:
                base_name_to_instance_names[base_name] = []
            base_name_to_instance_names[base_name].append(instance_name)

        return base_name_to_instance_names

    def _build_object_specs_lookup(self) -> dict[int, tuple[SceneMotif, ObjectSpec]]:
        """Build lookup dictionary for object specifications."""
        all_object_specs: dict[int, tuple[SceneMotif, ObjectSpec]] = {}
        for scene_motif in self.scene_motifs:
            for obj_spec in scene_motif.object_specs:
                if obj_spec.id:  # Check if ID exists and is valid
                    try:
                        all_object_specs[int(obj_spec.id)] = (scene_motif, obj_spec)
                    except (ValueError, TypeError):
                        print(f"Warning: Skipping object spec with non-integer ID {obj_spec.id} in motif {scene_motif.id}")

        print(f"Built lookup with {len(all_object_specs)} entries.")
        return all_object_specs

    async def _process_constrained_small_objects(
        self,
        constrained_small_objects: list,
        constrained_parent_motifs: set,
        all_object_specs: dict[int, tuple[SceneMotif, ObjectSpec]],
        output_dir: str,
        cfg,
        model
    ) -> set[str]:
        """Process constrained small objects for motifs that have specific assignments."""
        processed_motifs: set[str] = set()

        # Find motifs that have constrained small objects
        constrained_motifs = []
        for motif in self.scene_motifs:
            parent_object_ids = {spec.id for spec in motif.object_specs}
            if parent_object_ids.intersection(constrained_parent_motifs):
                constrained_motifs.append(motif)

        print(f"Found {len(constrained_motifs)} motifs with constrained small objects: {[m.id for m in constrained_motifs]}")

        for motif in constrained_motifs:
            # Get all small objects for parents in this motif
            parent_object_ids = {spec.id for spec in motif.object_specs}
            relevant_small_specs: list[ObjectSpec] = [
                obj for obj in constrained_small_objects
                if obj.parent_object in parent_object_ids
            ]

            if not relevant_small_specs:
                print(f"Motif {motif.id} has no relevant small objects")
                continue

            # Construct parent_name_to_id_map for the current motif
            constrained_motif_parent_name_to_id_map = self._build_parent_name_to_id_map(motif, constrained_parent_motifs)

            # Collect and process surface data
            layer_data = await self._collect_constrained_surface_data(
                motif, constrained_motif_parent_name_to_id_map, constrained_parent_motifs, output_dir
            )

            if not layer_data:
                print(f"Warning: No layer data collected for motif {motif.id}. Skipping constrained small object processing.")
                continue

            # Process small objects for this motif
            try:
                await self._process_single_motif_constrained_small_objects(
                    motif, relevant_small_specs, constrained_motif_parent_name_to_id_map,
                    layer_data, output_dir, cfg, model, all_object_specs
                )
                processed_motifs.add(motif.id)
            except Exception as e:
                print(f"Skipping motif {motif.id} small objects constrained population due to error: \n {e}\n")
                import traceback
                traceback.print_exc()
                continue

        return processed_motifs

    async def _collect_constrained_surface_data(
        self, motif: SceneMotif, parent_name_to_id_map: dict[str, int],
        constrained_parent_motifs: set, output_dir: str
    ) -> Optional[dict]:
        """Collect surface data for constrained small objects."""
        # Get instance names for constrained parents only
        constrained_parent_instance_names = [
            name for name, obj_id in parent_name_to_id_map.items()
            if obj_id in constrained_parent_motifs
        ]

        if not constrained_parent_instance_names:
            return None

        # Collect surface data for constrained parents
        layer_data, _, layer_fig = collect_surface_data(
            large_object_names=constrained_parent_instance_names,
            motif=motif,
            output_dir=output_dir,
            try_ransac=False
        )

        if layer_fig:
            plt.close(layer_fig)

        # Remap layer_data keys to use instance names
        if layer_data:
            remapped_layer_data = {}
            base_name_to_instance_names = self._group_instances_by_base_name(parent_name_to_id_map)

            # Create case-insensitive mapping for layer data keys
            layer_data_case_map = {key.lower(): key for key in layer_data.keys()}
            base_name_case_map = {name.lower(): name for name in base_name_to_instance_names.keys()}

            # Remap layer data using instance names with case-insensitive matching
            for layer_key, layer_info in layer_data.items():
                # Try exact match first
                if layer_key in base_name_to_instance_names:
                    instance_names = base_name_to_instance_names[layer_key]
                    for instance_name in instance_names:
                        if parent_name_to_id_map.get(instance_name) in constrained_parent_motifs:
                            remapped_layer_data[instance_name] = layer_info
                # Fall back to case-insensitive match
                elif layer_key.lower() in base_name_case_map:
                    actual_base_name = base_name_case_map[layer_key.lower()]
                    instance_names = base_name_to_instance_names[actual_base_name]
                    for instance_name in instance_names:
                        if parent_name_to_id_map.get(instance_name) in constrained_parent_motifs:
                            remapped_layer_data[instance_name] = layer_info

            return remapped_layer_data

        return None

    async def _process_single_motif_constrained_small_objects(
        self, motif: SceneMotif, relevant_small_specs: list,
        constrained_motif_parent_name_to_id_map: dict[str, int],
        layer_data: dict, output_dir: str, cfg, model, all_object_specs: dict
    ) -> None:
        """Process constrained small objects for a single motif."""
        small_obj_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))

        # Prepare layer_data for VLM using clean_layer_info
        layer_data_for_llm = clean_layer_info(layer_data)

        # Get layered structure for these constrained small objects
        parent_names_str = [
            name for name, obj_id in constrained_motif_parent_name_to_id_map.items()
            if obj_id in {obj.parent_object for obj in relevant_small_specs}
        ]

        objects_response_str = small_obj_session.send_with_validation(
            "small_objects_layered",
            {
                "small_objects": [obj.to_gpt_dict() for obj in relevant_small_specs],
                "motif_description": str(motif.description),
                "large_furniture": str(parent_names_str),
                "room_type": self.room_description,
                "layer_info": json.dumps(layer_data_for_llm)
            },
            lambda resp: validate_small_object_response(resp,
                                                      relevant_small_specs,
                                                      parent_names_str,
                                                      layer_data),
            json=True,
            verbose=True
        )

        parsed_llm_data = json.loads(extract_json(objects_response_str))
        added_small_objects_spec_container = self.scene_spec.add_multi_parent_small_objects(
            parsed_llm_data,
            constrained_motif_parent_name_to_id_map
        )

        relevant_small_specs = added_small_objects_spec_container.small_objects

        if not relevant_small_specs:
            print(f"Warning: No small objects were ultimately added to scene_spec for motif {motif.id} from the VLM response.")
            return
        else:
            print(f"Added {len(relevant_small_specs)} constrained small ObjectSpecs to scene_spec for motif {motif.id}.")

        if not relevant_small_specs:
            print(f"Warning: No small objects found in updated scene spec for motif {motif.id}")
            return
        else:
            print(f"Found {len(relevant_small_specs)} small objects to process for motif {motif.id}")

        analysis_data = {}
        analysis_str = small_obj_session.send_with_validation(
            "populate_surface_motifs",
            {
                "room_type": self.room_description,
                "small_objects": [obj.to_gpt_dict() for obj in relevant_small_specs],
            },
            lambda response: validate_arrangement_smc(response,
                                                    [obj.id for obj in relevant_small_specs],
                                                    relevant_small_specs,
                                                    enforce_same_layer=True,
                                                    enforce_same_surface=True),
            json=True,
            verbose=True,
        )
        analysis_data = json.loads(extract_json(analysis_str))

        if not cfg.mode.use_scene_motifs:
            analysis_str = create_individual_scene_motifs_with_analysis(
                [obj.to_dict() for obj in relevant_small_specs], analysis_data
            )
            analysis_data = json.loads(analysis_str)

        # Add height limits from layer data
        for arrangement in analysis_data["arrangements"]:
            if "furniture" in arrangement["composition"]:
                obj_ids = [item["id"] for item in arrangement["composition"]["furniture"]]
                arrangement_objs = [obj for obj in relevant_small_specs if obj.id in obj_ids]

                if arrangement_objs:
                    parent_id = arrangement_objs[0].parent_object
                    layer_key = arrangement_objs[0].placement_layer

                    if parent_id and layer_key:
                        parent_name = None
                        for spec in motif.object_specs:
                            if spec.id == parent_id:
                                parent_name = spec.name
                                break

                        if (layer_data and parent_name and
                            parent_name in layer_data and layer_key in layer_data[parent_name]):
                            layer_info = layer_data[parent_name][layer_key]
                            arrangement["composition"]["height_limit"] = round(layer_info.get("space_above", 0), 4)

        # Process and place small objects
        await self._process_and_place_small_objects(
            cfg, analysis_data, motif, relevant_small_specs, layer_data,
            output_dir, model, all_object_specs, True
        )

    def _get_remaining_motifs_for_unconstrained_processing(self, processed_motifs: set[str]) -> set[SceneMotif]:
        """Get motifs that still need unconstrained small object processing."""
        # Calculate remaining motifs, excluding both processed motifs and those with existing small objects
        all_eligible_motifs: set[SceneMotif] = set(filter_motifs_by_types(self.scene_motifs, [ObjectType.LARGE, ObjectType.WALL]))
        motifs_with_small_objects: set[SceneMotif] = set()

        # Find motifs that already have small objects populated
        for motif in all_eligible_motifs:
            for obj in motif.objects:
                if hasattr(obj, 'child_motifs') and obj.child_motifs:
                    motifs_with_small_objects.add(motif)
                    break

        # Convert processed_motifs (set of IDs) to set of motifs for set operations
        processed_motif_objects: set[SceneMotif] = {motif for motif in all_eligible_motifs if motif.id in processed_motifs}
        current_remaining_motifs: set[SceneMotif] = all_eligible_motifs - processed_motif_objects - motifs_with_small_objects

        print(f"Total eligible motifs: {len(all_eligible_motifs)}")
        print(f"Processed motifs (constrained): {len(processed_motifs)} - {list(processed_motifs)}")
        print(f"Motifs with existing small objects: {len(motifs_with_small_objects)} - {[m.id for m in motifs_with_small_objects]}")
        print(f"Remaining {len(current_remaining_motifs)} motifs for unconstrained small objects: {[m.id for m in current_remaining_motifs]}")

        return current_remaining_motifs

    async def populate_small_objects(self, cfg, output_dir: str, model=None) -> Optional[Session]:
        """
        Two-phase small object population:
        1. First process motifs that have specifically assigned small objects
        2. Then process remaining motifs with unconstrained small objects
        """
        if not self._setup_small_objects_population():
            return None

        constrained_small_objects, constrained_parent_motifs = self._get_constrained_small_objects()
        all_object_specs = self._build_object_specs_lookup()

        # Track which motifs have been processed (by ID)
        processed_motifs: set[str] = set()

        # --- Process Constrained Small Objects ---
        if constrained_small_objects:
            processed_motifs = await self._process_constrained_small_objects(
                constrained_small_objects, constrained_parent_motifs,
                all_object_specs, output_dir, cfg, model
            )
        
        # Process unconstrained small objects
        return await self._process_unconstrained_small_objects(
            cfg, processed_motifs, all_object_specs, output_dir, model
        )

    async def _process_unconstrained_small_objects(
        self, cfg, processed_motifs: set[str],
        all_object_specs: dict[int, tuple[SceneMotif, ObjectSpec]],
        output_dir: str, model
    ) -> Optional[Session]:
        """Process unconstrained small objects in iterations."""
        # Calculate remaining motifs, excluding both processed motifs and those with existing small objects
        remaining_motifs = self._get_remaining_motifs_for_unconstrained_processing(processed_motifs)

        if "small" not in cfg.mode.extra_types:
            print("Skipping unconstrained small objects from config.")
            return None

        ############## Unconstrained small objects with iterations ###################
        max_iterations = cfg.parameters.small_object_generation.max_iterations
        target_saturation = cfg.parameters.small_object_generation.target_occupancy_percent / 100.0
        iteration = 0
        small_obj_session = None

        # Track cumulative occupancy across all surfaces and iterations
        cumulative_occupancy = 0.0
        total_surfaces_processed = 0

        print(f"Starting unconstrained small object iterations (max: {max_iterations}, target saturation: {target_saturation * 100:.1f}%)")


        while iteration < max_iterations:
            print(f"\n{'='*60}")
            print(f"Unconstrained Small Objects - Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")

            # Recalculate remaining motifs for each iteration (as done in original code)
            remaining_motifs = self._get_remaining_motifs_for_unconstrained_processing(processed_motifs)

            if not remaining_motifs:
                print(f"No eligible motifs for iteration {iteration + 1}. Stopping iterations.")
                break

            print(f"Iteration {iteration + 1}: Processing {len(remaining_motifs)} motifs")
            print(f"Motif IDs: {[m.id for m in remaining_motifs]}")

            # Process one iteration of unconstrained small objects
            processed_motifs, iteration_session, motifs_processed, iteration_occupancy_data = await self._process_unconstrained_small_objects_iteration(
                cfg=cfg,
                remaining_motifs=remaining_motifs,
                processed_motifs=processed_motifs,
                all_object_specs=all_object_specs,
                output_dir=output_dir,
                model=model,
                iteration=iteration
            )

            # Update session for return value
            if iteration_session:
                small_obj_session = iteration_session

            # Accumulate occupancy data from this iteration
            if iteration_occupancy_data:
                iteration_occupancy = iteration_occupancy_data.get('average_occupancy', 0.0)
                surfaces_in_iteration = iteration_occupancy_data.get('surfaces_processed', 0)
                
                if surfaces_in_iteration > 0:
                    # Calculate weighted average occupancy across all iterations
                    cumulative_occupancy = ((cumulative_occupancy * total_surfaces_processed) +
                                          (iteration_occupancy * surfaces_in_iteration)) / (total_surfaces_processed + surfaces_in_iteration)
                    total_surfaces_processed += surfaces_in_iteration

            occupancy_ratio = cumulative_occupancy if total_surfaces_processed > 0 else 0.0
            print(f"Current occupancy ratio: {occupancy_ratio:.1f}")
            print(f"Motifs processed so far: {len(processed_motifs)}/{len(self.scene_motifs)}")

            # Check if we've reached target saturation or processed no new motifs
            if occupancy_ratio >= target_saturation:
                print(f"Target saturation reached: {occupancy_ratio:.1f} >= {target_saturation:.1f}")
                break
            elif motifs_processed == 0:
                print(f"No new motifs processed in iteration {iteration + 1}. Stopping iterations.")
                break
            elif iteration_occupancy_data and iteration_occupancy_data.get('surfaces_processed', 0) == 0:
                print(f"No surfaces processed in iteration {iteration + 1}. Stopping iterations.")
                break

            iteration += 1

        print(f"Completed unconstrained small object iterations. Total iterations: {iteration + 1}")
        print(f"Final occupancy ratio: {occupancy_ratio:.1f}")
        print(f"Total surfaces processed: {total_surfaces_processed}")

        # TODO: Current implementation has the following limitations:
        # 1. It does NOT check all objects across the entire scene simultaneously
        # 2. It does NOT process objects individually based on their current state
        # 3. It does NOT re-evaluate objects that were already processed in previous iterations
        #
        # This means that once a motif is processed, the system doesn't go back to check
        # if individual objects within that motif need additional small objects, even if
        # there might be available space.


    async def _process_unconstrained_small_objects_iteration(
        self,
        cfg: DictConfig,
        remaining_motifs: set[SceneMotif],
        processed_motifs: set[str],
        all_object_specs: dict[int, tuple[SceneMotif, ObjectSpec]],
        output_dir: str,
        model=None,
        iteration: int = 0
    ) -> tuple[set[str], Session, int, dict]:
        """
        Process one iteration of unconstrained small object population.

        Args:
            cfg: Configuration object
            remaining_motifs: Set of motifs available for unconstrained processing
            processed_motifs: Set of motif IDs already processed
            all_object_specs: Lookup dictionary for object specifications
            output_dir: Output directory path
            model: ModelManager instance
            iteration: Current iteration number

        Returns:
            Tuple of (updated_processed_motifs, session, motifs_processed_count, occupancy_data)
            where occupancy_data contains:
            - 'average_occupancy': weighted average occupancy ratio from DFS solver
            - 'surfaces_processed': number of surfaces processed in this iteration
        """
        motifs_processed_count = 0
        small_obj_session = None

        # Track occupancy data from DFS solver
        total_occupancy = 0.0
        total_surfaces = 0

        print(f"\nStarting unconstrained small object iteration {iteration}")
        print(f"Processing {len(remaining_motifs)} motifs for unconstrained small objects")

        motifs_skipped_no_specs = 0
        motifs_skipped_no_surfaces = 0
        motifs_processed_successfully = 0

        for motif in tqdm(remaining_motifs, desc=f"Iteration {iteration}: Processing Unconstrained Motifs"):


            # Skip motifs without object specs or already processed motifs
            if not motif.object_specs:
                print(f"Warning: No object specs found for motif {motif.id}. " +
                      "Skipping unconstrained small object population for this motif.")
                motifs_skipped_no_specs += 1
                continue

            ids: dict[str, str] = {}
            existing_objects, _ = motif.get_objects_by_names()
            for obj in existing_objects:
                try:
                    ids[obj.name] = obj.get_mesh_id()
                except Exception as e:
                    print(f"Error retrieving mesh id for object {obj.name if 'name' in locals() else obj}: {e}")

            has_support_surfaces = any(check_support_json_exists(id) for id in ids.values())


            if not has_support_surfaces:
                # if all ids not found, skip motif, else store object names and ids for later
                print(f"Warning: No support surface data found for any objects in motif {motif.id}. " +
                      "Skipping unconstrained small object population for this motif.")
                motifs_skipped_no_surfaces += 1
                continue
            # if all ids found, store object names and ids for later
            else:
                object_names: list[str] = list(ids.keys())
                # object_ids: list[str] = list(ids.values())

            print(f"Found {len(ids)} objects with support surface data for motif {motif.id}")
            motifs_processed_successfully += 1

            anchor_scene_objects, _ = find_anchor_object(motif, object_names)

            if not anchor_scene_objects:
                print(f"No SceneObjects identified as anchors within motif '{motif.id}'. " +
                      "Skipping unconstrained small object population for this motif.")
                continue

            # Extract names from the found anchor SceneObjects for the current motif
            current_motif_anchor_names = [aso.name for aso in anchor_scene_objects if hasattr(aso, 'name') and aso.name]

            # Collect surface data for this motif using the identified anchor_names from the current motif
            layer_data, _, layer_fig = collect_surface_data(
                large_object_names=current_motif_anchor_names,
                motif=motif, # Pass the current motif
                output_dir=output_dir,
                try_ransac=False
            )

            if layer_fig:
                plt.close(layer_fig)

            # Skip when no support-surface data is available – unless these small objects were explicitly constrained
            if not layer_data:
                print(f"Warning: No surface data collected for anchors in motif {motif.id}. Anchor names attempted: {current_motif_anchor_names}")
                continue # Skip this motif if no surface data for its anchors

            # Construct parent_name_to_id_map for the current motif
            current_motif_parent_name_to_id_map = self._build_parent_name_to_id_map(motif)

            # Remap layer_data keys to use instance names instead of base names
            if layer_data:
                remapped_layer_data = {}
                base_name_to_instance_names = {}

                # Group instance names by base name (batch lookup for efficiency)
                base_name_to_instance_names = self._group_instances_by_base_name(current_motif_parent_name_to_id_map)

                # Create case-insensitive mapping for layer data keys
                layer_data_case_map = {key.lower(): key for key in layer_data.keys()}
                base_name_case_map = {name.lower(): name for name in base_name_to_instance_names.keys()}

                # Remap layer data using instance names with case-insensitive matching
                for layer_key, layer_info in layer_data.items():
                    # Try exact match first
                    if layer_key in base_name_to_instance_names:
                        instance_names = base_name_to_instance_names[layer_key]
                        for instance_name in instance_names:
                            remapped_layer_data[instance_name] = layer_info
                    # Fall back to case-insensitive match
                    elif layer_key.lower() in base_name_case_map:
                        actual_base_name = base_name_case_map[layer_key.lower()]
                        instance_names = base_name_to_instance_names[actual_base_name]
                        for instance_name in instance_names:
                            remapped_layer_data[instance_name] = layer_info

                layer_data = remapped_layer_data

            small_obj_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))

            # Collect information about existing objects on each parent
            existing_objects_info = self._collect_existing_objects_info(motif, current_motif_parent_name_to_id_map)

            # Debug: Print existing objects information
            print(f"DEBUG: Existing objects info for motif {motif.id}: {existing_objects_info}")

            # Get additional small objects for this motif with layer-specific placement
            prompt_data = {
                "small_objects": "***Suggest small objects for this motif***",
                "motif_description": str(motif.description),
                "large_furniture": str(list(current_motif_parent_name_to_id_map.keys())), # Use instance names instead of base names
                "room_type": self.room_type,
                "layer_info": json.dumps(clean_layer_info(layer_data))
            }

            # Add existing objects information if available
            if existing_objects_info and any(existing_objects_info.values()):
                prompt_data["existing_objects"] = json.dumps(existing_objects_info)
                print(f"DEBUG: Added existing_objects to prompt: {existing_objects_info}")
                print(f"DEBUG: Full prompt_data: {prompt_data}")
            else:
                prompt_data["existing_objects"] = "No existing small objects on these surfaces"
                print("DEBUG: No existing objects found, using default message")
                print(f"DEBUG: Full prompt_data: {prompt_data}")

            objects_response = small_obj_session.send(
                "small_objects_layered",
                prompt_data,
                json=True,
                verbose=True
            )
            layered_response = json.loads(extract_json(objects_response))
            # No filtering needed - the VLM should respect existing objects in the prompt
            filtered_response = layered_response

            all_newly_added_small_objects_for_motif = []

            # Iterate through the instance names from the parent_name_to_id_map
            for instance_name, parent_id in current_motif_parent_name_to_id_map.items():
                parent_spec = self.scene_spec.get_object_by_id(parent_id)
                if not parent_spec:
                    print(f"Warning: Could not find ObjectSpec for parent ID {parent_id} (instance '{instance_name}') in motif '{motif.id}'. Skipping this parent.")
                    continue

                print(f"Processing parent instance: '{instance_name}' (ID: {parent_id}) within motif '{motif.id}'.")

                # Check if the VLM response has suggestions for this specific instance name
                llm_response_key_for_parent = None
                for key_from_llm in filtered_response.keys():
                    if key_from_llm.lower() == instance_name.lower():
                        llm_response_key_for_parent = key_from_llm
                        break

                if llm_response_key_for_parent and isinstance(filtered_response.get(llm_response_key_for_parent), dict):
                    sub_response_for_parent = {llm_response_key_for_parent: filtered_response[llm_response_key_for_parent]}

                    # self.scene_spec is mutated by add_layered_objects_from_response (via self.add_objects)
                    # The returned 'added_spec' contains only the new ObjectSpecs from this specific call.
                    added_spec_for_this_parent = self.scene_spec.add_layered_objects_from_response(
                        sub_response_for_parent,
                        parent_id, # This is the ID of the ObjectSpec within the current motif
                        instance_name # This is the instance name for this specific parent
                    )

                    if added_spec_for_this_parent and added_spec_for_this_parent.small_objects:
                        all_newly_added_small_objects_for_motif.extend(added_spec_for_this_parent.small_objects)
                    else:
                        print(f"  Warning: No new small objects added by add_layered_objects_from_response for parent instance '{instance_name}' (ID: {parent_id}).")
                else:
                    print(f"  Warning: No specific suggestions found in VLM response for parent instance '{instance_name}' (ID: {parent_id}) in motif '{motif.id}', or response format invalid. VLM response keys: {list(filtered_response.keys())}")

            # After iterating all parent instances for the current motif:
            relevant_small_specs = all_newly_added_small_objects_for_motif

            if not relevant_small_specs:
                print(f"Warning: No small objects found in updated scene spec for motif {motif.id}")
                continue

            print(f"Found {len(relevant_small_specs)} small objects to process for motif {motif.id}")

            # group small objects to motifs by parent_object
            analysis_data = {}
            analysis_str = small_obj_session.send_with_validation(
                "populate_surface_motifs",
                {
                    "room_type": self.room_description,
                    "small_objects": [obj.to_gpt_dict() for obj in relevant_small_specs],
                },
                lambda response: validate_arrangement_smc(response,
                                                        [obj.id for obj in relevant_small_specs],
                                                        relevant_small_specs,
                                                        enforce_same_layer=True,
                                                        enforce_same_surface=True),
                json=True,
                verbose=True,
            )
            analysis_data = json.loads(extract_json(analysis_str))

            if not cfg.mode.use_scene_motifs:
                analysis_str = create_individual_scene_motifs_with_analysis([obj.to_dict() for obj in relevant_small_specs], analysis_data)
                analysis_data = json.loads(analysis_str)

            # Add a height limit to the analysis for each arrangement from the layer_data
            for arrangement in analysis_data["arrangements"]:
                # Extract parent object and layer information from the objects in this arrangement
                if "furniture" in arrangement["composition"]:
                    obj_ids = [item["id"] for item in arrangement["composition"]["furniture"]]
                    # Find the corresponding objects
                    arrangement_objs = [obj for obj in relevant_small_specs if obj.id in obj_ids]

                    if arrangement_objs:
                        # Get parent and layer info from the first object (assuming all have same parent/layer)
                        parent_id = arrangement_objs[0].parent_object
                        layer_key = arrangement_objs[0].placement_layer

                        if parent_id and layer_key:
                            # Get parent name
                            parent_name = None
                            for spec in motif.object_specs:
                                if spec.id == parent_id:
                                    parent_name = spec.name
                                    break

                            # Add height limit from layer data
                            if layer_data and parent_name and parent_name in layer_data and layer_key in layer_data[parent_name]:
                                layer_info = layer_data[parent_name][layer_key]
                                arrangement["composition"]["height_limit"] = round(layer_info.get("space_above", 0), 4)

            try:
                # Process and place these small objects
                motif_occupancy = await self._process_and_place_small_objects(cfg, analysis_data, motif, relevant_small_specs, layer_data, output_dir, model, all_object_specs)

                # Accumulate occupancy data from this motif
                if motif_occupancy and isinstance(motif_occupancy, dict):
                    total_occupancy += motif_occupancy.get('total_occupancy', 0.0)
                    total_surfaces += motif_occupancy.get('surfaces_processed', 0)

                processed_motifs.add(motif.id)
                motifs_processed_count += 1
            except Exception as e:
                print(f"Skipping motif {motif.id} small objects unconstrained population due to error: \n {e}\n")
                import traceback
                traceback.print_exc()
                continue

        print(f"Completed unconstrained small object iteration {iteration}. Processed {motifs_processed_count} motifs.")
        print(f"DEBUG: Iteration {iteration} summary:")
        print(f"  - Total motifs input: {len(remaining_motifs)}")
        print(f"  - Motifs processed successfully: {motifs_processed_successfully}")
        print(f"  - Motifs skipped (no specs): {motifs_skipped_no_specs}")
        print(f"  - Motifs skipped (no surfaces): {motifs_skipped_no_surfaces}")
        print(f"  - Total occupancy accumulated: {total_occupancy}")
        print(f"  - Total surfaces processed: {total_surfaces}")

        # Prepare occupancy data for return
        iteration_occupancy_data = {
            'average_occupancy': total_occupancy / total_surfaces if total_surfaces > 0 else 0.0,
            'surfaces_processed': total_surfaces,
            'total_occupancy': total_occupancy
        }

        return processed_motifs, small_obj_session, motifs_processed_count, iteration_occupancy_data

    async def _process_and_place_small_objects(self, cfg, analysis, motif: SceneMotif,
                                               object_specs: list[ObjectSpec], layer_data: dict, output_dir: str,
                                               model=None,
                                               all_object_specs: dict = {},
                                               solver_fallback=False):
        """Helper method to process and place small objects for a motif"""
        if not analysis:
            print(f"Warning: No analysis provided for motif {motif.id}")
            return {}
            
        try:
            layout_data = analysis

            print("="*100)
            print(f"Analysis with height limits: {layout_data}")
            print("="*100)

            # Track occupancy data for this motif
            motif_occupancy_data = {
                'average_occupancy': 0.0,
                'surfaces_processed': 0,
                'total_occupancy': 0.0
            }

            # Create support surface constraints for small objects
            support_surface_constraints = self._create_support_surface_constraints(
                object_specs, layer_data, motif
            )
            
            # Generate initial small motifs from layout data
            initial_small_motifs, _ = await process_scene_motifs(
                object_specs, 
                layout_data, 
                output_dir, 
                self.room_description, 
                model,
                object_type=ObjectType.SMALL,
                support_surface_constraints=support_surface_constraints
            )
            
            print(f"Generated {len(initial_small_motifs)} initial small motifs for {motif.id}: {[m.id for m in initial_small_motifs]}")
            
            if not initial_small_motifs:
                print(f"Warning: No small motifs generated for motif {motif.id}")
                return
                
            # Deduplicate small motifs by ID to prevent duplicates
            deduplicated_motifs = {}
            for sm in initial_small_motifs:
                if sm.id not in deduplicated_motifs:
                    deduplicated_motifs[sm.id] = sm
            
            initial_small_motifs = list(deduplicated_motifs.values())
            print(f"After deduplication: {len(initial_small_motifs)} small motifs for {motif.id}")
            
            # Group small motifs by parent object
            parent_to_motifs = {}
            processed_small_motif_ids = set()
            
            for small_motif in initial_small_motifs:
                # Skip if this small motif has already been processed
                if small_motif.id in processed_small_motif_ids:
                    continue
                    
                processed_small_motif_ids.add(small_motif.id)
                
                # Find the parent_id for this small motif
                parent_ids = set()
                for spec in small_motif.object_specs:
                    if spec.parent_object:
                        parent_ids.add(spec.parent_object)
                
                # Add this motif to each parent's list (only once per parent)
                for parent_id in parent_ids:
                    if parent_id not in parent_to_motifs:
                        parent_to_motifs[parent_id] = []
                    # Only add if not already in this parent's list    
                    if not any(m.id == small_motif.id for m in parent_to_motifs[parent_id]):
                        parent_to_motifs[parent_id].append(small_motif)
            
            print(f"Grouped small motifs by parent: {[(pid, [m.id for m in motifs]) for pid, motifs in parent_to_motifs.items()]}")

            # Process each parent object using the PASSED dictionary
            for parent_id, relevant_motifs in parent_to_motifs.items():
                # Check if the parent_id corresponds to an object within the `motif` argument.
                # The `motif` argument is the primary large object motif context (e.g., floating_wall_shelf).
                parent_spec_in_current_motif = None
                for spec_in_main_motif in motif.object_specs: # `motif` is the one passed to _process_and_place_small_objects
                    if spec_in_main_motif.id == parent_id:
                        parent_spec_in_current_motif = spec_in_main_motif
                        break
                
                target_motif: SceneMotif
                parent_spec: ObjectSpec

                if parent_spec_in_current_motif:
                    # The parent object is within the current processing motif. Use this motif directly.
                    target_motif = motif 
                    parent_spec = parent_spec_in_current_motif
                    # print(f"DEBUG: Using current motif '{target_motif.id}' as parent for object ID {parent_id} ({parent_spec.name}).")
                else:
                    # Fallback to the global map if the parent_id isn't directly in the current `motif`'s specs.
                    # This might happen if parent_id refers to an object in a different motif.
                    if not self.scene_spec:
                        print(f"Warning: SceneSpec not available for global lookup of parent ID {parent_id}.")
                        continue
                    parent_info_from_map = all_object_specs.get(parent_id)
                    if not parent_info_from_map:
                        print(f"Warning: Parent object ID {parent_id} not found in current motif '{motif.id}' or global lookup.")
                        continue
                    target_motif, parent_spec = parent_info_from_map
                    # print(f"DEBUG: Using globally looked-up motif '{target_motif.id}' for parent object ID {parent_id} ({parent_spec.name}).")
                
                print(f"Processing parent {parent_spec.name} (id: {parent_id}) for {len(relevant_motifs)} small motifs")
                # Get layout suggestions first
                layout_suggestions, layer_data, layer_fig = populate_furniture(
                    large_object_names=[parent_spec.name],
                    room_desc=self.room_description,
                    generated_small_motif=relevant_motifs,
                    motif=target_motif
                )
                                
                # Skip when no support-surface data is available – unless these small objects were explicitly constrained
                if not layer_data:
                    print(f"Warning: No layer data found for {parent_spec.name}")
                    continue
                
                # Get the constrained layout for this parent
                parent_occupancy_data, constrained_layout = optimize_small_objects(
                    cfg,
                    fig=target_motif.fig,
                    layout_suggestions=layout_suggestions,
                    output_path=os.path.join(output_dir, f"{target_motif.id}", "solver"),
                    layer_fig=layer_fig,
                    layer_data=layer_data,
                    small_motifs=relevant_motifs,
                    fallback=solver_fallback
                )

                # Accumulate occupancy data from this parent
                if parent_occupancy_data:
                    motif_occupancy_data['total_occupancy'] += parent_occupancy_data.get('total_occupancy', 0.0)
                    motif_occupancy_data['surfaces_processed'] += parent_occupancy_data.get('surfaces_processed', 0)

                updated_motifs = update_small_motifs_from_constrained_layout(
                    constrained_layout, relevant_motifs, target_motif, layer_data
                )
                
                print(f"Got {len(updated_motifs)} updated small motifs for {parent_spec.name}")

                # Find the actual parent object and add child motifs to it
                parent_object = target_motif.get_object(parent_spec.name)
                if parent_object:
                    if updated_motifs:
                        print(f"Adding {len(updated_motifs)} small motifs to parent {parent_object.name}")
                        parent_object.child_motifs.extend(updated_motifs)
                        self.add_motifs(updated_motifs)
                else:
                    print(f"Warning: Could not find parent object {parent_spec.name} in motif {motif.id}")
                
                # Run spatial optimization for the small objects after placement
                if updated_motifs and cfg.mode.get('enable_spatial_optimization', False):
                    try:
                        from hsm_core.scene.processing_helpers import run_spatial_optimization_for_stage, filter_motifs_needing_optimization
                        motifs_needing_optimization = filter_motifs_needing_optimization(updated_motifs)
                        if motifs_needing_optimization:
                            run_spatial_optimization_for_stage(
                                scene=self,
                                cfg=cfg,
                                current_stage_motifs=motifs_needing_optimization,
                                object_type=ObjectType.SMALL,
                                output_dir=Path(output_dir),
                                stage_name=f"parent_{parent_spec.name}"
                            )
                    except Exception as e:
                        print(f"Warning: Spatial optimization failed for small objects on {parent_spec.name}: {e}")
                        import traceback
                        traceback.print_exc()

            # Calculate average occupancy for this motif
            if motif_occupancy_data['surfaces_processed'] > 0:
                motif_occupancy_data['average_occupancy'] = (
                    motif_occupancy_data['total_occupancy'] / motif_occupancy_data['surfaces_processed']
                )

            return motif_occupancy_data

        except Exception as e:
            print(f"Error processing small objects for motif {motif.id}: {e}")
            import traceback
            traceback.print_exc()

    def _collect_existing_objects_info(self, motif: SceneMotif, parent_name_to_id_map: dict) -> dict:
        """
        Collect information about existing small objects on each parent object.

        Args:
            motif: The motif containing parent objects
            parent_name_to_id_map: Mapping from instance names to object IDs

        Returns:
            dict: Information about existing objects on each parent
        """
        existing_info = {}

        print(f"DEBUG: Collecting existing objects for motif {motif.id}")
        print(f"DEBUG: Parent name to ID map: {parent_name_to_id_map}")
        print(f"DEBUG: Motif objects: {[obj.name if hasattr(obj, 'name') else str(obj) for obj in motif.objects]}")

        # First, let's look at ALL motifs in the scene to find existing small objects
        # The existing objects might be in separate motifs, not as child_motifs
        all_scene_motifs = self.scene_motifs  # Access all motifs in the scene

        print(f"DEBUG: Total motifs in scene: {len(all_scene_motifs)}")
        for sm in all_scene_motifs:
            print(f"DEBUG: Scene motif {sm.id} (type: {sm.object_type}) has {len(sm.objects)} objects:")
            for obj in sm.objects:
                obj_name = obj.name if hasattr(obj, 'name') else 'unnamed'
                placement_data = obj.placement_data if hasattr(obj, 'placement_data') else None
                print(f"    - {obj_name}: placement_data={placement_data}")

        for instance_name, parent_id in parent_name_to_id_map.items():
            print(f"DEBUG: Looking for parent {instance_name} with ID {parent_id}")

            # Find the parent object in the motif - try multiple ways
            parent_obj = None

            # Method 1: Try by ID if objects have ID attribute
            for obj in motif.objects:
                if hasattr(obj, 'id') and obj.id == parent_id:
                    parent_obj = obj
                    print(f"DEBUG: Found parent by ID: {obj.name if hasattr(obj, 'name') else 'unnamed'}")
                    break

            # Method 2: Try by name matching (instance_name might be like "nightstand_1")
            if not parent_obj:
                for obj in motif.objects:
                    if hasattr(obj, 'name'):
                        # Check if instance_name contains the object name or vice versa
                        if (instance_name in obj.name or
                            obj.name in instance_name or
                            instance_name.split('_')[0] == obj.name):
                            parent_obj = obj
                            print(f"DEBUG: Found parent by name matching: {obj.name}")
                            break

            if not parent_obj:
                print(f"DEBUG: No parent object found for {instance_name} (ID: {parent_id})")
                continue

            print(f"DEBUG: Checking parent object: {parent_obj.name if hasattr(parent_obj, 'name') else 'unnamed'}")
            print(f"DEBUG: Parent has child_motifs: {hasattr(parent_obj, 'child_motifs')}")
            if hasattr(parent_obj, 'child_motifs'):
                print(f"DEBUG: Number of child motifs: {len(parent_obj.child_motifs) if parent_obj.child_motifs else 0}")

            # Look for existing objects in two ways:
            # 1. As child_motifs on the parent object (current approach)
            # 2. As separate motifs in the scene that are associated with this parent

            existing_objects = []

            # Method 1: Check child_motifs (current approach)
            if hasattr(parent_obj, 'child_motifs') and parent_obj.child_motifs:
                print(f"DEBUG: Found {len(parent_obj.child_motifs)} child motifs on parent {instance_name}")
                for child_motif in parent_obj.child_motifs:
                    print(f"DEBUG: Processing child motif: {child_motif.id if hasattr(child_motif, 'id') else 'no-id'}")
                    if hasattr(child_motif, 'object_specs') and child_motif.object_specs:
                        for obj_spec in child_motif.object_specs:
                            existing_objects.append({
                                'name': obj_spec.name,
                                'type': obj_spec.object_type.value if hasattr(obj_spec, 'object_type') else 'unknown'
                            })
                            print(f"DEBUG: Found existing object via child_motif: {obj_spec.name}")

            # Method 2: Look for small object motifs that might be associated with this parent
            # Use ObjectSpec parent_object relationship as the primary method
            parent_name = parent_obj.name if hasattr(parent_obj, 'name') else instance_name
            print(f"DEBUG: Looking for small object motifs with parent_id={parent_id} or parent_name='{parent_name}'")

            for scene_motif in all_scene_motifs:
                if scene_motif.object_type == ObjectType.SMALL:
                    print(f"DEBUG: Checking small motif {scene_motif.id}")
                    parent_match = False
                    
                    # Method A: Check ObjectSpec parent_object relationships (most reliable)
                    if scene_motif.object_specs:
                        for obj_spec in scene_motif.object_specs:
                            if hasattr(obj_spec, 'parent_object') and obj_spec.parent_object == parent_id:
                                parent_match = True
                                print(f"DEBUG: Found parent match via ObjectSpec.parent_object: {obj_spec.parent_object} == {parent_id}")
                                break
                    
                    # Method B: Check placement_data (for constrained objects)
                    if not parent_match and hasattr(scene_motif, 'placement_data') and scene_motif.placement_data:
                        parent_ref = scene_motif.placement_data.get('parent_object')
                        if parent_ref and (parent_ref == parent_name or parent_ref.lower() == parent_name.lower()):
                            parent_match = True
                            print(f"DEBUG: Found parent match via placement_data: {parent_ref} matches {parent_name}")
                    
                    # Method C: Check parent_id (legacy fallback)
                    if not parent_match and hasattr(scene_motif, 'parent_id') and scene_motif.parent_id:
                        try:
                            motif_parent_id = int(scene_motif.parent_id)
                            if motif_parent_id == parent_id:
                                parent_match = True
                                print(f"DEBUG: Found parent match via motif parent_id: {motif_parent_id} == {parent_id}")
                        except (ValueError, TypeError):
                            pass
                    
                    # If we found a parent match, add all objects from this small motif
                    if parent_match:
                        for small_obj in scene_motif.objects:
                            existing_objects.append({
                                'name': small_obj.name if hasattr(small_obj, 'name') else f"{scene_motif.id}_obj",
                                'type': 'small_object'
                            })
                            print(f"DEBUG: Found existing object: {small_obj.name if hasattr(small_obj, 'name') else f'{scene_motif.id}_obj'}")

            if existing_objects:
                existing_info[instance_name] = existing_objects
                print(f"DEBUG: Added {len(existing_objects)} existing objects for {instance_name}")
                for obj in existing_objects:
                    print(f"  - {obj['name']} ({obj['type']})")
            else:
                print(f"DEBUG: No existing objects found for {instance_name}")

        print(f"DEBUG: Final existing info: {existing_info}")
        return existing_info



    def _create_support_surface_constraints(
        self,
        small_objects: list[ObjectSpec],
        layer_data: dict,
        parent_motif: SceneMotif
    ) -> dict[str, dict]:
        """
        Create support surface constraints for small objects based on their parent objects and layer data.
        
        Args:
            small_objects: List of small object specs that need support surface constraints
            layer_data: Dictionary containing layer information for parent objects
            parent_motif: Parent motif containing the parent objects
            
        Returns:
            dict[str, dict]: Mapping of object labels to their support surface constraints
        """
        constraints = {}
        
        # Create lookup for parent objects in the motif
        parent_objects = {}
        for obj in parent_motif.objects:
            if hasattr(obj, 'name'):
                parent_objects[obj.name] = obj
        
        for small_obj in small_objects:
            if not hasattr(small_obj, 'name') or not hasattr(small_obj, 'parent_object'):
                continue
                
            # Find parent object info
            parent_id = small_obj.parent_object
            parent_spec = None
            for spec in parent_motif.object_specs:
                if spec.id == parent_id:
                    parent_spec = spec
                    break
            
            if not parent_spec:
                continue
                
            parent_name = parent_spec.name
            placement_layer = getattr(small_obj, 'placement_layer', None)
            placement_surface = getattr(small_obj, 'placement_surface', None)
            
            # Get layer information for this parent and layer
            if layer_data and parent_name in layer_data and placement_layer in layer_data[parent_name]:
                layer_info = layer_data[parent_name][placement_layer]
                
                # Build constraint dict
                surface_constraints = {}

                if placement_surface is not None and 'surfaces' in layer_info:
                    # Surface-level constraints
                    for surface in layer_info['surfaces']:
                        if surface.get('surface_id') == placement_surface:
                            w = surface.get('width', 1.0)
                            d = surface.get('depth', 1.0)
                            area = surface.get('area', w * d)
                            bounds = surface.get('bounds', {'width': w, 'depth': d})
                            surface_constraints = {
                                'available_area': area,
                                'bounds': bounds,
                                'max_height': layer_info.get('space_above', 0.5),
                                'parent_name': parent_name,
                                'layer': placement_layer,
                                'surface_id': placement_surface,
                            }
                            break
                else:
                    # Layer-level fallback (no explicit surface)
                    w = 0.8
                    d = 0.8
                    area = w * d
                    surface_constraints = {
                        'available_area': area,
                        'bounds': {'width': w, 'depth': d},
                        'max_height': layer_info.get('space_above', 0.5),
                        'parent_name': parent_name,
                        'layer': placement_layer,
                    }
                
                if surface_constraints:
                    constraints[small_obj.name.lower()] = surface_constraints
                    print(f"Created support surface constraints for {small_obj.name}: {surface_constraints}")
        
        return constraints

    def get_motifs_by_types(self, object_types: list[ObjectType]|ObjectType) -> Optional[list['SceneMotif']]:
        """Get motifs by their types."""
        if isinstance(object_types, ObjectType):
            return self._filter_motifs_by_type(self.scene_motifs, object_types)
        else:
            return [motif for motif in self.scene_motifs if motif.object_type in object_types]

    def _validate_door_location(self, door_location: tuple[float, float]) -> tuple[float, float]:
        """
        Validate that the door location is at least 0.5m away from any corner
        to ensure a 1m wide door can fit properly.
        
        Returns:
            Valid door location or adjusted location if too close to corners
        """        
        door = Cutout(
            location=door_location,
            cutout_type="door",
            width=DOOR_WIDTH,
            height=DOOR_HEIGHT,
        )
        
        # Initial validation attempt
        if door.validate(self.room_polygon):
            return door.location

        # If initial validation fails, print warning and try to adjust to the closest wall
        print(f"Warning: Initial door location {door_location} is invalid. Attempting to adjust to the closest wall.")
        if door.adjust_to_wall(self.room_polygon):  # Pass None for existing_cutouts as this is the first door
            # adjust_to_wall internally calls validate again. If it returns True, the door.location is updated and valid.
            print(f"Door successfully adjusted to a valid position on the closest wall: {door.location}")
            return door.location
        else:
            # If adjust_to_wall also fails, then proceed with the original fallback (longest wall midpoint)
            print(f"Warning: Could not find valid door location for {door_location} even after trying to adjust to the closest wall. Falling back to longest wall midpoint.")
            # Fallback to center of longest wall
            longest_wall = None
            max_length = 0
            
            for i in range(len(self.room_vertices)):
                p1 = self.room_vertices[i]
                p2 = self.room_vertices[(i + 1) % len(self.room_vertices)]
                length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                
                if length > max_length and length >= DOOR_WIDTH + 0.2:
                    max_length = length
                    longest_wall = (p1, p2)
            
            if longest_wall:
                mid_x = (longest_wall[0][0] + longest_wall[1][0]) / 2
                mid_y = (longest_wall[0][1] + longest_wall[1][1]) / 2
                print(f"Door moved to longest wall midpoint ({mid_x:.2f}, {mid_y:.2f})")
                return (mid_x, mid_y)
            
            # use original location (which is known to be invalid at this point, but it's the final fallback)
            print(f"Warning: All fallback methods failed. Using original invalid door location {door_location}.")
            return door_location

    def _validate_window_location(self, window_location: list[tuple[float, float]]|None) -> list[tuple[float, float]]:
        """
        Validate that window locations are properly positioned on walls and don't overlap
        with each other or the door.

        Args:
            window_location: List of (x, y) coordinates for windows

        Returns:
            List of validated window locations
        """
        if not window_location:
            return []

        # Setup door as existing cutout
        door = self._create_door_cutout()
        all_cutouts = [door] if door.is_valid else []
        validated_windows = []

        # Process each window
        for window_pos in window_location:
            window = self._create_window_cutout(window_pos)
            if self._try_validate_window(window, all_cutouts):
                validated_windows.append(window.location)
                all_cutouts.append(window)
            else:
                # Try alternative placement
                alt_window = self._find_alternative_window_placement(all_cutouts)
                if alt_window:
                    validated_windows.append(alt_window.location)
                    all_cutouts.append(alt_window)

        return validated_windows

    def _create_door_cutout(self) -> Cutout:
        """Create and validate door cutout."""
        door = Cutout(
            location=self.door_location if hasattr(self, 'door_location') else (0, 0),
            cutout_type="door",
        )
        door.validate(self.room_polygon)
        return door

    def _create_window_cutout(self, location: tuple[float, float]) -> Cutout:
        """Create window cutout with standard dimensions."""
        return Cutout(
            location=location,
            cutout_type="window",
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            bottom_height=WINDOW_BOTTOM_HEIGHT,
        )

    def _try_validate_window(self, window: Cutout, all_cutouts: list) -> bool:
        """Try to validate window, with fallback to wall adjustment."""
        if window.validate(self.room_polygon, all_cutouts):
            return True
        return window.adjust_to_wall(self.room_polygon, all_cutouts)

    def _find_alternative_window_placement(self, all_cutouts: list) -> Optional[Cutout]:
        """Find alternative placement for window on available walls."""
        for j in range(len(self.room_vertices)):
            if self._wall_has_both_cutouts(j, all_cutouts):
                continue

            wall_length = self._calculate_wall_length(j)
            if wall_length < WINDOW_WIDTH + 0.2:
                continue

            mid_point = self._calculate_wall_midpoint(j)
            alt_window = Cutout(
                location=mid_point,
                cutout_type="window",
                width=min(WINDOW_WIDTH, wall_length * 0.6)
            )

            if self._try_validate_window(alt_window, all_cutouts):
                return alt_window

        return None

    def _wall_has_both_cutouts(self, wall_index: int, all_cutouts: list) -> bool:
        """Check if wall already has both door and window."""
        has_door = any(c.cutout_type == "door" and c.closest_wall_index == wall_index for c in all_cutouts)
        has_window = any(c.cutout_type == "window" and c.closest_wall_index == wall_index for c in all_cutouts)
        return has_door and has_window

    def _calculate_wall_length(self, wall_index: int) -> float:
        """Calculate length of wall segment."""
        p1 = self.room_vertices[wall_index]
        p2 = self.room_vertices[(wall_index + 1) % len(self.room_vertices)]
        return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5

    def _calculate_wall_midpoint(self, wall_index: int) -> tuple[float, float]:
        """Calculate midpoint of wall segment."""
        p1 = self.room_vertices[wall_index]
        p2 = self.room_vertices[(wall_index + 1) % len(self.room_vertices)]
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _filter_motifs_by_type(self, motifs: list[SceneMotif], object_type: Optional[ObjectType] = None) -> list[SceneMotif]:
        """Filter motifs by object type if specified."""
        if object_type:
            return [m for m in motifs if m.object_type == object_type]
        return motifs

    def get_motifs_needing_optimization(self, object_type: Optional[ObjectType] = None) -> list[SceneMotif]:
        """
        Get motifs that need spatial optimization.

        Args:
            object_type: Optional filter by object type

        Returns:
            list[SceneMotif]: Motifs that need optimization
        """
        motifs = self._filter_motifs_by_type(self._scene_motifs, object_type)
        return [m for m in motifs if not getattr(m, 'is_spatially_optimized', False)]

    def get_optimized_motifs(self, object_type: Optional[ObjectType] = None) -> list[SceneMotif]:
        """
        Get motifs that are already spatially optimized.

        Args:
            object_type: Optional filter by object type

        Returns:
            list[SceneMotif]: Motifs that are already optimized
        """
        motifs = self._filter_motifs_by_type(self._scene_motifs, object_type)
        return [m for m in motifs if hasattr(m, 'is_spatially_optimized') and m.is_spatially_optimized]

    def get_scene_optimization_status(self) -> dict[str, Any]:
        """
        Get optimization status for all motifs in the scene.
        
        Returns:
            dict: Comprehensive optimization status
        """
        status = {
            "total_motifs": len(self._scene_motifs),
            "optimized_motifs": 0,
            "unoptimized_motifs": 0,
            "motifs_by_type": {},
            "motif_details": []
        }
        
        for motif in self._scene_motifs:
            if hasattr(motif, 'get_optimization_status'):
                motif_status = motif.get_optimization_status()
                status["motif_details"].append(motif_status)
                
                if motif_status["is_spatially_optimized"]:
                    status["optimized_motifs"] += 1
                else:
                    status["unoptimized_motifs"] += 1
                
                obj_type = motif_status["object_type"]
                if obj_type not in status["motifs_by_type"]:
                    status["motifs_by_type"][obj_type] = {"total": 0, "optimized": 0}
                status["motifs_by_type"][obj_type]["total"] += 1
                if motif_status["is_spatially_optimized"]:
                    status["motifs_by_type"][obj_type]["optimized"] += 1
        
        return status

    def get_all_motifs(self) -> list[SceneMotif]:
        """Get all motifs in the scene, including nested child motifs."""
        return _get_all_motifs_recursive(self._scene_motifs)

    def get_all_objects(self) -> list[SceneObject]:
        """Get all scene objects from all motifs, including nested ones."""
        all_objects = []
        seen_ids = set()
        for motif in self.get_all_motifs():
            for obj in motif.objects:
                if obj.id not in seen_ids:
                    all_objects.append(obj)
                    seen_ids.add(obj.id)
        return all_objects


