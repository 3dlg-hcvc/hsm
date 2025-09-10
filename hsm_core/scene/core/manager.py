from __future__ import annotations
from hsm_core.scene.core.objects import SceneObject
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING, Dict
import trimesh

if TYPE_CHECKING:
    from hsm_core.scene.core.objects import SceneObject
from hsm_core.scene.geometry.room_geometry import create_custom_room
from hsm_core.scene.validation.placement import validate_door_location, validate_window_location
from hsm_core.scene.utils.utils import load_scene_state
from hsm_core.scene.core.motif import SceneMotif
from hsm_core.scene.geometry.placer import SceneObjectPlacer
from hsm_core.scene.core.spec import SceneSpec
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.setup.scene_creation import create_scene_from_motifs
from hsm_core.constants import *
from hsm_core.utils import get_logger


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
            window_location: Optional list of (x, y) tuples for window locations. Defaults to None (no windows).
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
        self.door_location = validate_door_location(self, door_location)
        self.window_location = validate_window_location(self, window_location)

        self.scene: Optional[trimesh.Scene] = None
        self.scene_placer: Optional[SceneObjectPlacer] = None
        self.scene_spec: Optional[SceneSpec] = scene_spec
        self.enable_spatial_optimization = enable_spatial_optimization

        self._mesh_cache: Dict[str, trimesh.Trimesh] = {}
        self._scene_motifs: list[SceneMotif] = []
        if scene_motifs:
            self.add_motifs(scene_motifs)
    
    def add_motifs(self, motifs: list[SceneMotif]) -> None:
        existing_ids: set = {motif.id for motif in self._scene_motifs}
        # TODO: Handle duplicate motifs
        for m in motifs:
            if m.id in existing_ids:
                logger = get_logger('scene.manager')
                logger.warning(f"Motif {m.id} already exists in scene")
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
        # Clear mesh cache since object positions may have changed
        self.invalidate_mesh_cache()

    def get_or_load_mesh(self, obj: 'SceneObject') -> Optional[trimesh.Trimesh]:
        """
        Get a mesh from cache or load it if not cached.

        Args:
            obj: SceneObject to get mesh for

        Returns:
            Preprocessed trimesh mesh or None if loading fails
        """
        if obj.name not in self._mesh_cache:
            from hsm_core.scene.utils.mesh_utils import preprocess_object_mesh
            mesh = preprocess_object_mesh(obj, verbose=False)
            if mesh is not None:
                self._mesh_cache[obj.name] = mesh
                return mesh
            return None
        return self._mesh_cache[obj.name]

    def invalidate_mesh_cache(self) -> None:
        """Clear the shared mesh cache."""
        self._mesh_cache.clear()

    def get_mesh_cache_size(self) -> int:
        """Get the number of meshes currently cached."""
        return len(self._mesh_cache)
    
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

    def create_scene(self) -> None:
        """
        Create a trimesh scene from the current state.

        This method generates the 3D scene and scene_placer from the current motifs.
        The scene is cached and will persist until motifs are modified or invalidate_scene() is called.
        """
        if not self._scene_motifs:
            self._create_empty_scene()
            return
        
        door_loc = self.door_location or (0, 0)
        top_level_motifs = self._scene_motifs
        enable_spatial_optimization = getattr(self, 'enable_spatial_optimization', True)

        logger = get_logger('scene.manager')
        logger.info(f"Creating 3D scene from {len(top_level_motifs)} scene motifs")

        window_location = None
        if self.window_location and isinstance(self.window_location, list):
            window_location = [(float(x), float(y)) for x, y in self.window_location]

        self.scene, self.scene_placer = create_scene_from_motifs(
            scene_motifs=top_level_motifs,
            room_polygon=self.room_polygon,
            door_location=door_loc,
            window_location=window_location,
            floor_height=0.0,  # Floor is at z=0
            room_height=self.room_height,
            enable_spatial_optimization=enable_spatial_optimization,
            scene_manager=self  # Pass self for scene properties access
        )

        if self.scene and self.scene_placer:
            logger.info("=" * 50)
            logger.info(f"Final scene contains {len(self.scene_placer.placed_objects)} meshes \n")

    def get_all_objects(self) -> list[SceneObject]:
        """Get all scene objects from all motifs, including nested ones."""
        from hsm_core.scene.utils.motif_utils import get_all_objects
        return get_all_objects(self)

    def get_motifs_by_types(self, object_types: list[ObjectType] | ObjectType) -> Optional[list[SceneMotif]]:
        """Get motifs by their types."""
        from hsm_core.scene.utils.motif_utils import get_motifs_by_types
        return get_motifs_by_types(self, object_types)

    async def populate_small_objects(self, cfg, output_dir: str, vis_output_dir, model=None):
        """Populate small objects in the scene."""
        from hsm_core.scene.processing.small_object_placer import populate_small_objects
        return await populate_small_objects(self, cfg, output_dir, vis_output_dir, model)

    def save(self, output_dir: Path, suffix: str = "", recreate_scene: bool = False, save_scene_state: bool = False) -> None:
        """Save the scene to the specified output directory."""
        from hsm_core.scene.io.export import save_scene
        save_scene(self, output_dir, suffix, recreate_scene, save_scene_state)

    def _create_empty_scene(self) -> None:
        """
        Create a minimal scene with just room geometry when no motifs are available.
        """
        from hsm_core.scene.geometry.placer import SceneObjectPlacer

        self.scene_placer = SceneObjectPlacer(self.room_polygon, self.room_height)
        door_loc = self.door_location or (0, 0)
        window_loc = self.window_location if isinstance(self.window_location, list) and self.window_location else None

        self.scene_placer.create_room_geom(
            room_polygon=self.room_polygon,
            door_location=door_loc,
            window_location=window_loc
        )

        self.scene = self.scene_placer.scene
    
        
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