"""
Scene Spatial Optimizer

This optimizer uses actual meshes and room geometry for collision detection
and support validation.
"""

import copy
import time
import logging
from typing import List, Dict, Optional, Tuple, Set, TYPE_CHECKING, Union
import numpy as np
from logging import Logger
import trimesh
import trimesh.transformations
from pathlib import Path

from hsm_core.scene.scene_3d import SceneObjectPlacer
from hsm_core.scene.objects import SceneObject
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.solvers.config import SceneSpatialOptimizerConfig

if TYPE_CHECKING:
    from hsm_core.scene.manager import Scene

logger: Logger = logging.getLogger(__name__)

class SceneSpatialOptimizer:
    """
    Mesh-based spatial optimizer for scene objects.

    Uses trimesh collision detection and raycasting for precise spatial optimization.
    Supports both individual object and motif-level optimization with type-aware logic.
    """
    
    def __init__(self, scene: 'Scene', config: Optional[SceneSpatialOptimizerConfig] = None):
        """
        Initialize the mesh-based spatial optimizer.
        
        Args:
            scene: Scene object containing motifs and room geometry.
            config: Configuration for spatial optimization.
        """
        self.scene = scene
        self.config = config or SceneSpatialOptimizerConfig()
        
        # Mesh caches for performance
        self._object_meshes: Dict[str, trimesh.Trimesh] = {}
        self._base_object_meshes: Dict[str, trimesh.Trimesh] = {}  # Immutable base meshes
        self._room_scene: Optional[trimesh.Scene] = None
        self._floor_mesh: Optional[trimesh.Trimesh] = None
        self._wall_meshes: List[trimesh.Trimesh] = []
        self._ceiling_mesh: Optional[trimesh.Trimesh] = None
        
        self._collision_manager_pool: Dict[str, trimesh.collision.CollisionManager] = {}
        
        # Statistics
        self.stats = {
            'objects_processed': 0,
            'collisions_resolved': 0,
            'support_fixes_applied': 0,
            'processing_time': 0.0,
            'early_exits': 0,
        }
        
    def optimize_objects(self, objects: List[SceneObject], context_objects: Optional[List[SceneObject]] = None) -> List[SceneObject]:
        """
        Optimize scene objects using mesh-based collision detection and support validation.

        Args:
            objects: Objects to optimize.
            context_objects: Context objects for collision/support validation.

        Returns:
            Optimized SceneObject instances.
        """
        if not objects:
            logger.debug("No objects to optimize - returning early")
            return objects

        if context_objects is None:
            context_objects = []

        start_time = time.time()

        if self.config.debug_output:
            logger.info("Starting spatial optimization for %d objects", len(objects))
            logger.debug("Context objects: %d", len(context_objects))
        
        # Initialize room geometry and object meshes
        self._initialize_room_geometry()
        self._load_object_meshes(objects, context_objects)
        
        # Choose optimization approach based on configuration
        if self.config.use_motif_level_optimization:
            optimized_objects = self._optimize_by_motifs(objects)
        else:
            optimized_objects = self._optimize_individually(objects)

        # Update statistics
        self.stats['processing_time'] = time.time() - start_time
        self._print_summary()
        
        if self.config.debug_output:
            logger.info("Optimization complete - processed %d objects", len(optimized_objects))
        
        return optimized_objects

    def _initialize_room_geometry(self) -> None:
        """Initialize room geometry from the scene for mesh-based operations."""

        temp_scene_placer = SceneObjectPlacer(room_height=self.scene.room_height)
        
        try:
            room_geometry = temp_scene_placer.create_room_geom(self.scene.room_polygon, self.scene.door_location, self.scene.window_location)

            self._floor_mesh = room_geometry['floor']
            self._wall_meshes = [room_geometry[name] for name in room_geometry.keys() if name.startswith('wall')]

            # Create ceiling mesh if room height is available
            if hasattr(self.scene, 'room_height') and self.scene.room_height:
                self._create_ceiling_mesh()
                        
            for name, mesh in room_geometry.items():
                if name.startswith('wall'):
                    self._wall_meshes.append(mesh)
                elif name == 'floor':
                    self._floor_mesh = mesh
            
            logger.info("Room geometry initialized for scene spatial optimization")

        except Exception as e:
            logger.error("Failed to initialize room geometry: %s", e)
            self._room_scene = None

    def _create_ceiling_mesh(self) -> None:
        """Create ceiling mesh based on room polygon and height."""
        try:
            if not hasattr(self.scene, 'room_polygon') or not self.scene.room_polygon:
                return

            if self._floor_mesh is None:
                return
            ceiling_mesh = self._floor_mesh.copy()
            
            # Translate to ceiling height
            ceiling_transform = trimesh.transformations.translation_matrix([0, self.scene.room_height, 0])
            ceiling_mesh.apply_transform(ceiling_transform)
            
            # Flip normals to point downward
            ceiling_mesh.faces = np.fliplr(ceiling_mesh.faces)
            
            self._ceiling_mesh = ceiling_mesh
            logger.debug("Ceiling mesh created for collision detection")
            
        except Exception as e:
            logger.error("Failed to create ceiling mesh: %s", e)

    def _load_object_meshes(self, objects: List[SceneObject], context_objects: Optional[List[SceneObject]] = None) -> None:
        """Load trimesh objects for all scene objects including context objects."""
        all_objects = objects + (context_objects or [])
        for obj in all_objects:
            try:
                if obj.name not in self._object_meshes:
                    mesh = self._load_single_object_mesh(obj)
                    if mesh is not None:
                        # Store working mesh
                        self._object_meshes[obj.name] = mesh.copy()
                        # Store immutable base mesh if caching is enabled
                        if self.config.enable_mesh_caching:
                            self._base_object_meshes[obj.name] = mesh.copy()
            except Exception as e:
                logger.warning("Failed to load mesh for %s: %s", obj.name, e)

    def _load_single_object_mesh(self, obj: SceneObject) -> Optional[trimesh.Trimesh]:
        """Load and preprocess mesh for a single object."""
        try:
            from hsm_core.scene.scene_3d import preprocess_object_mesh
            # Check if mesh_path is set
            if not obj.mesh_path or obj.mesh_path.strip() == "":
                logger.warning("Object '%s' has no mesh_path set - cannot load mesh for spatial optimization", obj.name)
                return None
            
            # Load and preprocess the mesh
            mesh = preprocess_object_mesh(obj)
            if mesh is None:
                logger.warning("Failed to load mesh for '%s' from path '%s' - mesh loading returned None", obj.name, obj.mesh_path)
                return None
            
            # Apply object's rotation
            if obj.rotation != 0:
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    angle=np.radians(obj.rotation),
                    direction=[0, 1, 0])
                mesh.apply_transform(rotation_matrix)
                
            # Apply object's position
            translation_matrix = trimesh.transformations.translation_matrix(obj.position)
            mesh.apply_transform(translation_matrix)
            
            return mesh
            
        except Exception as e:
            logger.warning("Exception loading mesh for '%s': %s", obj.name, e)
            return None

    def _optimize_by_motifs(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Optimize objects grouped by motifs, preserving internal relationships."""
        if self.config.debug_output:
            logger.info("Optimizing %d objects by motifs", len(objects))
        
        optimized_objects: list[SceneObject] = []
        motif_groups = self._group_objects_by_motif(objects)
        
        # Optimize each motif as a unit
        for motif_id, motif_objects in motif_groups.items():
            if self.config.debug_output:
                logger.info("Optimizing motif: %s (%d objects)", motif_id, len(motif_objects))
            
            context_objects = [obj for obj in objects if obj not in motif_objects]
            optimized_motif_objects = self._optimize_motif_as_unit(motif_objects, context_objects)
            optimized_objects.extend(optimized_motif_objects)
            
            # ------------------------------------------------------------------
            # Replace the original motif objects inside the *shared* `objects`
            # list so that subsequent motif iterations use the **updated**
            # world positions as context.  Without this, later motifs may test
            # collisions/support against stale coordinates and apply erroneous
            # corrections (e.g. lowering a pot that was already fixed).
            # ------------------------------------------------------------------
            for orig, new in zip(motif_objects, optimized_motif_objects):
                try:
                    idx = objects.index(orig)
                    objects[idx] = new
                except ValueError:
                    # Should not happen, but fail-safe: append
                    objects.append(new)
        
        return optimized_objects

    def _optimize_individually(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Optimize each object individually."""
        if self.config.debug_output:
            logger.info("Optimizing %d objects individually", len(objects))
        
        optimized_objects = []
        for obj in objects:
            context_objects = [o for o in objects if o != obj]
            optimized_obj = self._optimize_single_object(obj, context_objects)
            optimized_objects.append(optimized_obj)
        
        return optimized_objects

    def _optimize_motif_as_unit(self, motif_objects: List[SceneObject], context_objects: List[SceneObject]) -> List[SceneObject]:
        """
        Optimize an entire motif as a single unit, preserving internal object relationships.
        Use minimal adjustments to maintain DFS solver's valid placements.
        """
        if not motif_objects:
            return []

        needs_optimisation = False
        for obj in motif_objects:
            if (
                self._find_mesh_collisions(obj, context_objects)
                or not self._is_properly_supported_mesh(obj, context_objects)
            ):
                needs_optimisation = True
                break

        if not needs_optimisation:
            # Motif is already well-placed, preserve it
            if self.config.debug_output:
                motif_id = motif_objects[0].motif_id if hasattr(motif_objects[0], 'motif_id') else 'unknown'
                logger.debug("Motif %s is already well-placed", motif_id)
            
            # Return the objects with optimized_world_pos set to their current positions
            # to maintain consistency with the optimization interface
            preserved_motif_objects: List[SceneObject] = []
            for obj in motif_objects:
                optimized_obj = copy.deepcopy(obj)
                # Set optimized_world_pos to current position (no change)
                optimized_obj.position = (float(obj.position[0]), float(obj.position[1]), float(obj.position[2]))
                optimized_obj.optimized_world_pos = (float(obj.position[0]), float(obj.position[1]), float(obj.position[2]))
                preserved_motif_objects.append(optimized_obj)
            
            return preserved_motif_objects
        
        # Create combined mesh for the motif
        combined_mesh = self._create_combined_motif_mesh(motif_objects)
        if combined_mesh is None:
            # Fallback to individual optimization with minimal adjustments
            return [self._optimize_single_object(obj, context_objects + [o for o in motif_objects if o != obj]) 
                   for obj in motif_objects]
        
        # Create a representative object for the motif
        motif_representative = self._create_motif_representative(motif_objects, combined_mesh)
        
        # Cache the combined mesh so collision resolution can find it
        self._object_meshes[motif_representative.name] = combined_mesh.copy()
        
        # Optimize the representative object with minimal adjustments
        optimized_representative = self._optimize_single_object(motif_representative, context_objects)
        
        # Calculate the transformation applied to the motif using bottom-centered positions
        original_bottom_center = np.array(motif_representative.position, dtype=float)
        new_bottom_center = np.array(optimized_representative.position, dtype=float)
        translation = new_bottom_center - original_bottom_center
        
        # Apply maximum translation limit to prevent large motif movements
        # max_translation = self.config.max_motif_translation
        # translation_magnitude = np.linalg.norm(translation)
        # if translation_magnitude > max_translation:
        #     translation = translation / translation_magnitude * max_translation
        #     if self.config.debug_output:
        #         print(f"    Limited motif translation to {max_translation}m (was {translation_magnitude:.3f}m)")
        
        # Apply the same transformation to all objects in the motif
        optimized_motif_objects: List[SceneObject] = []

        for obj in motif_objects:
            # Work on a shallow copy so we don't accidentally mutate the caller
            optimized_obj = copy.deepcopy(obj)
            world_pos_arr = np.array(obj.position, dtype=float) + np.array(translation, dtype=float)
            world_pos = (float(world_pos_arr[0]), float(world_pos_arr[1]), float(world_pos_arr[2]))

            # store position as well for small motif downstream use
            optimized_obj.position = world_pos
            optimized_obj.optimized_world_pos = world_pos

            optimized_motif_objects.append(optimized_obj)

        if self.config.debug_output and np.linalg.norm(translation) > 0.001:
            logger.debug("Applied minimal translation %.3fm to motif (stored in optimized_world_pos)", np.linalg.norm(translation))

        return optimized_motif_objects

    def _create_combined_motif_mesh(self, motif_objects: List[SceneObject]) -> Optional[trimesh.Trimesh]:
        """Create a combined mesh representing the entire motif."""
        try:
            motif_meshes = []
            for obj in motif_objects:
                if obj.name in self._object_meshes:
                    motif_meshes.append(self._object_meshes[obj.name])
            
            if not motif_meshes:
                return None
            
            if len(motif_meshes) == 1:
                return motif_meshes[0].copy()
            else:
                return trimesh.util.concatenate(motif_meshes)
                
        except Exception as e:
            logger.warning("Failed to create combined motif mesh: %s", e)
            return None

    def _create_motif_representative(self, motif_objects: List[SceneObject], combined_mesh: trimesh.Trimesh) -> SceneObject:
        """Create a representative SceneObject for the entire motif."""
        # Calculate combined properties (bottom-centered)
        bounds = combined_mesh.bounds  # shape (2, 3): [min_xyz, max_xyz]
        dimensions = bounds[1] - bounds[0]
        bottom_center = (
            float((bounds[0][0] + bounds[1][0]) * 0.5),
            float(bounds[0][1]),
            float((bounds[0][2] + bounds[1][2]) * 0.5),
        )

        # Use the first object as a template
        template_obj = motif_objects[0]
        
        representative = SceneObject(
            name=f"motif_{template_obj.motif_id}_combined",
            position=bottom_center,
            dimensions=tuple(dimensions),
            rotation=template_obj.rotation,
            mesh_path=template_obj.mesh_path,
            obj_type=template_obj.obj_type,
            motif_id=template_obj.motif_id,
            parent_name=getattr(template_obj, "parent_name", None)
        )
        
        return representative

    def _optimize_single_object(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Optimize single object with collision resolution and support validation."""
        if self.config.debug_output:
            if obj.name.startswith("motif_") or "_combined" in obj.name:
                logger.debug("Optimizing motif: %s (%s)", obj.name, obj.obj_type.name)
            else:
                logger.debug("Optimizing single object: %s (%s)", obj.name, obj.obj_type.name)
        
        # Step 1: Resolve collisions
        collision_resolved_obj = self._resolve_mesh_collisions(obj, context_objects)
        
        # Step 2: Ensure proper support
        support_fixed_obj = self._ensure_mesh_support(collision_resolved_obj, context_objects)
        
        # Update statistics
        if collision_resolved_obj.position != obj.position:
            self.stats['collisions_resolved'] += 1
        if support_fixed_obj.position != collision_resolved_obj.position:
            self.stats['support_fixes_applied'] += 1
        
        self.stats['objects_processed'] += 1
        
        # --------------------------------------------------------------
        # Persist world-space coordinate so downstream code can consume
        # the optimiser output without relying on in-place mutation.
        # --------------------------------------------------------------
        support_fixed_obj.optimized_world_pos = (
            float(support_fixed_obj.position[0]),
            float(support_fixed_obj.position[1]),
            float(support_fixed_obj.position[2])
        )
        
        return support_fixed_obj

    def _resolve_mesh_collisions(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Resolve collisions using actual mesh intersection detection with minimal adjustments."""
        if self.config.debug_output:
            logger.debug("Resolving collisions for %s", obj.name)

        if obj.name not in self._object_meshes:
            logger.warning("No mesh found for %s - skipping collision resolution", obj.name)
            return obj  # Can't resolve without mesh
        
        obj_mesh = self._object_meshes[obj.name]
        max_iterations = self.config.max_collision_iterations
        step_size = self.config.get_step_size(obj.obj_type)
        
        current_obj = copy.deepcopy(obj)
        
        # First check if object actually has collisions
        initial_collisions = self._find_mesh_collisions(current_obj, context_objects)
        
        if not initial_collisions:
            # Object is already well-placed, don't adjust
            if self.config.debug_output:
                logger.debug("%s is already collision-free, preserving position", obj.name)
            return current_obj
        
        if self.config.debug_output:
            logger.debug("Initial collisions: %s", initial_collisions)
            logger.debug("Resolving %d collision(s) for %s", len(initial_collisions), obj.name)
        
        # Calculate initial penetration depth for early exit optimization
        initial_penetration_depth = self._calculate_penetration_depth(current_obj, initial_collisions)
        current_penetration_depth = initial_penetration_depth
        
        for iteration in range(max_iterations):
            # Check for collisions with context objects
            colliding_objects = self._find_mesh_collisions(current_obj, context_objects)
            
            # -------------------------------------------------------------
            # Ignore shallow collisions with the *parent* object for small
            # items.  Those contacts are expected (object resting on a table)
            # and should be handled by support validation instead of the
            # generic collision resolver.
            # -------------------------------------------------------------
            if (
                current_obj.obj_type == ObjectType.SMALL and
                colliding_objects and
                getattr(current_obj, "parent_name", None)
            ):
                parent_name = current_obj.parent_name
                parent_only = all(o.name == parent_name for o in colliding_objects)
                if parent_only:
                    pen_depth = self._calculate_penetration_depth(current_obj, colliding_objects)
                    if pen_depth < self.config.get_support_tolerance(current_obj.obj_type):
                        # Treat as no collision – rely on support logic instead
                        colliding_objects = []
            
            if not colliding_objects:
                if iteration > 0 and self.config.debug_output:
                    logger.debug("Collision resolved for %s after %d iteration(s)", obj.name, iteration + 1)
                break
            
            # Early exit optimization: check if penetration is minimal
            if self.config.enable_early_exit_optimization:
                current_penetration_depth = self._calculate_penetration_depth(current_obj, colliding_objects)
                if current_penetration_depth < self.config.penetration_depth_tolerance:
                    if self.config.debug_output:
                        logger.debug("Early exit for %s: penetration depth %.4fm < %.3fm",
                                   obj.name, current_penetration_depth, self.config.penetration_depth_tolerance)
                    self.stats['early_exits'] += 1
                    break
            else:
                current_penetration_depth = self._calculate_penetration_depth(current_obj, colliding_objects)
            
            # Prepare collision list for resolution
            collision_list: List[Union[SceneObject, str]] = list(colliding_objects)
            
            # Adaptive step size based on penetration depth
            adaptive_step_size = min(step_size, current_penetration_depth * self.config.adaptive_step_factor)
            
            # Exit if adaptive step size is too small to make meaningful progress
            if adaptive_step_size < self.config.min_step_size:
                if self.config.debug_output:
                    logger.debug("Exiting collision resolution for %s: adaptive step size %.6fm < minimum %.6fm",
                               obj.name, adaptive_step_size, self.config.min_step_size)
                break
            
            # Move away from collisions
            current_obj = self._resolve_single_mesh_collision(current_obj, collision_list, adaptive_step_size)
            
            # Update mesh position efficiently using cached transforms
            self._update_object_mesh_position_efficient(current_obj)
            
            # Track progress and reduce step size if no progress
            if current_penetration_depth >= initial_penetration_depth * self.config.step_reduction_threshold:
                step_size *= self.config.step_reduction_factor  # Gradually reduce step size
                
                # Exit if step size becomes too small to make meaningful progress
                if step_size < self.config.min_step_size:
                    if self.config.debug_output:
                        logger.debug("Exiting collision resolution for %s: step size %.6fm < minimum %.6fm",
                                   obj.name, step_size, self.config.min_step_size)
                    break
            
            initial_penetration_depth = current_penetration_depth
        
        return current_obj

    def _find_mesh_collisions(self, obj: SceneObject, context_objects: List[SceneObject]) -> List[SceneObject]:
        """Find collisions using mesh intersection detection."""
        if obj.name not in self._object_meshes:
            return []
        
        obj_mesh = self._object_meshes[obj.name]
        colliding = []
        
        # Get relevant collision context (type-aware)
        relevant_context = self._get_relevant_collision_context(obj, context_objects)
        
        for other_obj in relevant_context:
            if other_obj.name in self._object_meshes:
                other_mesh = self._object_meshes[other_obj.name]
                
                # Check mesh intersection
                collision_result = self._check_mesh_collision(
                    obj_mesh, other_mesh, return_penetration=True
                )
                
                if isinstance(collision_result, tuple):
                    is_collision, penetration = collision_result
                else:
                    is_collision = collision_result
                    penetration = 0.0
                
                if is_collision:
                    colliding.append(other_obj)
        
        return colliding

    def _check_mesh_collision(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, 
                             return_penetration: bool = False) -> Union[bool, Tuple[bool, float]]:
        """Check if two meshes are colliding, optionally return penetration depth."""
        try:
            # Create collision manager with reuse logic
            mesh1_id = id(mesh1)
            mesh2_id = id(mesh2)
            manager_key = f"{min(mesh1_id, mesh2_id)}_{max(mesh1_id, mesh2_id)}"
            
            if self.config.enable_collision_manager_cache and manager_key in self._collision_manager_pool:
                collision_manager = self._collision_manager_pool[manager_key]
            else:
                collision_manager = trimesh.collision.CollisionManager()
                collision_manager.add_object('obj1', mesh1)
                collision_manager.add_object('obj2', mesh2)
                if self.config.enable_collision_manager_cache:
                    self._collision_manager_pool[manager_key] = collision_manager

            if return_penetration:
                collision_result = collision_manager.in_collision_internal(return_data=True)
            else:
                collision_result = collision_manager.in_collision_internal()

            # Handle different return formats from trimesh
            if isinstance(collision_result, tuple):
                is_collision = bool(collision_result[0])
                contacts = collision_result[1] if len(collision_result) > 1 else None
            else:
                is_collision = bool(collision_result)
                contacts = None
            
            if not return_penetration:
                return is_collision
            
            # Calculate penetration depth if requested
            penetration_depth = 0.0
            if is_collision and contacts:
                for contact in contacts:
                    if hasattr(contact, 'depth') and contact.depth > 0:
                        penetration_depth = max(penetration_depth, contact.depth)
            
            return is_collision, penetration_depth
            
        except Exception as e:
            logger.warning("Mesh collision detection failed: %s", e)
            # Fallback to bounding box collision
            is_collision = self._check_bounding_box_collision(mesh1, mesh2)
            if return_penetration:
                return is_collision, 0.0
            return is_collision

    def _check_bounding_box_collision(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> bool:
        """Fallback bounding box collision detection."""
        bounds1 = mesh1.bounds
        bounds2 = mesh2.bounds
        
        # Check for overlap in all dimensions
        tolerance = self.config.collision_tolerance
        
        overlap_x = bounds1[1][0] > bounds2[0][0] - tolerance and bounds1[0][0] < bounds2[1][0] + tolerance
        overlap_y = bounds1[1][1] > bounds2[0][1] - tolerance and bounds1[0][1] < bounds2[1][1] + tolerance
        overlap_z = bounds1[1][2] > bounds2[0][2] - tolerance and bounds1[0][2] < bounds2[1][2] + tolerance
        
        return overlap_x and overlap_y and overlap_z

    def _resolve_single_mesh_collision(self, obj: SceneObject, colliding_objects: List[Union[SceneObject, str]], step_size: float) -> SceneObject:
        """
        Resolve a single collision by moving the object away with minimal adjustment.
        Priority: vertical movement (up) > minimal horizontal movement.
        Principle: Assume objects are mostly well-placed, only small adjustments needed.
        """
        resolved_obj = copy.deepcopy(obj)
        
        if not colliding_objects:
            return resolved_obj

        # For collisions, prioritize moving up (vertical) over horizontal movement
        # This prevents objects from being moved into other objects' spaces
        
        # Step 1: Try vertical movement first (pull object up slightly)
        # vertical_step = min(step_size * self.config.vertical_step_factor, self.config.max_vertical_adjustment)
        vertical_step = step_size * self.config.vertical_step_factor
        
        # Check if moving up resolves collisions
        test_obj = copy.deepcopy(resolved_obj)
        test_obj.position = (test_obj.position[0], test_obj.position[1] + vertical_step, test_obj.position[2])
        
        # Update mesh position for collision test
        self._update_object_mesh_position_efficient(test_obj)
        
        # Test if vertical movement resolves all collisions
        scene_objects_only = [c for c in colliding_objects if isinstance(c, SceneObject)]
        remaining_collisions = self._find_mesh_collisions(test_obj, scene_objects_only)
        
        if not remaining_collisions:
            # Vertical movement resolved the collision
            resolved_obj = test_obj
            if self.config.debug_output:
                logger.debug("Resolved collision for %s by moving up %.3fm", obj.name, vertical_step)
            return resolved_obj
        
        # Step 2: If vertical movement doesn't work, use minimal horizontal movement
        movement_direction = np.array([0.0, 0.0, 0.0])
        
        for colliding in colliding_objects:
            if isinstance(colliding, str) and colliding == "room_boundary":
                # Move toward room center with minimal adjustment
                if hasattr(self.scene, 'room_polygon') and self.scene.room_polygon:
                    room_centroid = self.scene.room_polygon.centroid
                    room_center = np.array([room_centroid.x, obj.position[1], room_centroid.y])
                    direction = room_center - np.array(obj.position)
                    direction[1] = 0  # Don't move vertically for room boundaries
                    if np.linalg.norm(direction) > 1e-6:
                        movement_direction += direction / np.linalg.norm(direction)
            elif isinstance(colliding, SceneObject):
                # Move away from colliding object with minimal adjustment
                direction = np.array(obj.position) - np.array(colliding.position)
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    movement_direction += direction
        
        if np.linalg.norm(movement_direction) > 1e-6:
            movement_direction = movement_direction / np.linalg.norm(movement_direction)
        else:
            # Fallback direction if no valid movement calculated
            movement_direction = np.array([1.0, 0.0, 0.0])
        
        # Apply movement constraints based on object type
        movement_direction = self._apply_movement_constraints(movement_direction, obj.obj_type)
        
        # Use much smaller horizontal step size for minimal adjustment
        # horizontal_step = min(step_size * self.config.horizontal_step_factor, self.config.max_horizontal_adjustment)
        horizontal_step = step_size * self.config.horizontal_step_factor
        
        # Apply movement
        new_position = np.array(obj.position) + movement_direction * horizontal_step
        resolved_obj.position = tuple(new_position)
        
        if self.config.debug_output:
            logger.debug("Adjusted %s horizontally by %.3fm", obj.name, horizontal_step)
        
        return resolved_obj

    def _ensure_mesh_support(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Ensure object is properly supported using mesh-based validation and fixes."""
        if not self._is_properly_supported_mesh(obj, context_objects):
            return self._fix_mesh_support(obj, context_objects)
        return obj

    def _is_properly_supported_mesh(self, obj: SceneObject, context_objects: List[SceneObject]) -> bool:
        """Check if an object is properly supported using mesh-based raycasting."""
        if obj.obj_type in [ObjectType.LARGE, ObjectType.SMALL]:
            return self._check_surface_support_mesh(obj, context_objects)
        elif obj.obj_type == ObjectType.WALL:
            return self._check_wall_attachment_mesh(obj)
        elif obj.obj_type == ObjectType.CEILING:
            return self._check_ceiling_attachment_mesh(obj)
        return True # Default to supported

    def _check_surface_support_mesh(self, obj: SceneObject, context_objects: List[SceneObject]) -> bool:
        """Check for surface support by raycasting down from the object's bottom vertices."""
        if self.config.debug_output:
            logger.debug("Surface support check for %s (%s)", obj.name, obj.obj_type.name)
            logger.debug("Position: %s, Dimensions: %s", obj.position, obj.dimensions)
        
        if obj.name not in self._object_meshes:
            if self.config.debug_output:
                logger.debug("No mesh loaded for %s, assuming supported", obj.name)
                logger.debug("Available meshes: %s", list(self._object_meshes.keys()))
            return True
        
        # Bottom of object in world coordinates (position is bottom-centered)
        object_bottom_y: float = obj.position[1]
        
        # Get support tolerance for all object types
        support_tolerance = self.config.get_support_tolerance(obj.obj_type)
        
        if self.config.debug_output:
            logger.debug("Object bottom Y: %.3fm, Support tolerance: %.3fm", object_bottom_y, support_tolerance)
        
        # Check against floor
        if self._floor_mesh and object_bottom_y < 0.1:  # If bottom is close to floor
            if self.config.debug_output:
                logger.debug("Checking floor support (object bottom: %.3fm)", object_bottom_y)
            
            # Create raycast from object's bottom center
            ray_origin = np.array([obj.position[0], object_bottom_y + 0.00001, obj.position[2]])
            ray_direction = np.array([0, -1, 0])
            
            try:
                locations, _, _ = self._floor_mesh.ray.intersects_location([ray_origin], [ray_direction])
                if len(locations) > 0:
                    # Check if floor is within reasonable distance
                    loc_arr = np.asarray(locations)
                    floor_distance = abs(float(loc_arr[0, 1]) - (object_bottom_y + 0.01))
                    if self.config.debug_output:
                        logger.debug("Floor raycast hit at distance: %.3fm", floor_distance)
                    if floor_distance < support_tolerance:
                        if self.config.debug_output:
                            logger.debug("%s is supported by floor", obj.name)
                        return True
                    elif self.config.debug_output:
                        logger.debug("Floor distance %.3fm > tolerance %.3fm", floor_distance, support_tolerance)
                elif self.config.debug_output:
                    logger.debug("Floor raycast found no hits")
            except Exception as e:
                if self.config.debug_output:
                    logger.debug("Floor raycast failed: %s", e)

        # Check against other objects
        support_context = self._get_relevant_support_context(obj, context_objects)
        if self.config.debug_output:
            logger.debug("Support context: %s (from %d total)", [s.name for s in support_context], len(context_objects))
        
        for sup_obj in support_context:
            if self.config.debug_output:
                logger.debug("Checking support from %s (%s)", sup_obj.name, sup_obj.obj_type.name)
                logger.debug("Support object position: %s, dimensions: %s", sup_obj.position, sup_obj.dimensions)
            
            supported, _ = self._compute_support_from_object(
                obj,
                sup_obj,
                object_bottom_y,
                support_tolerance,
            )

            if supported:
                if self.config.debug_output:
                    logger.debug("%s is supported by %s", obj.name, sup_obj.name)
                return True
            elif self.config.debug_output:
                logger.debug("%s does not support %s", sup_obj.name, obj.name)

        if self.config.debug_output:
            logger.debug("%s is NOT supported by any surface", obj.name)
            logger.debug("Checked %d support objects: %s", len(support_context), [s.name for s in support_context])
        return False

    def _check_wall_attachment_mesh(self, obj: SceneObject) -> bool:
        """Check if a wall object is attached to a wall using mesh raycasting."""
        wall_meshes = self._select_wall_meshes(obj)
        if obj.name not in self._object_meshes or not wall_meshes:
            if self.config.debug_output:
                logger.debug("%s: No wall meshes available, assuming proper attachment", obj.name)
            return True  # Cannot check, assume it's fine

        obj_mesh = self._object_meshes[obj.name]

        # Get the "front" direction vector based on object's rotation
        angle_rad = np.radians(obj.rotation)
        front_vector = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
        back_vector = -front_vector

        # Calculate potential back face positions to check
        check_points = []

        # Variant A: origin at back face (HSSD assets)
        check_points.extend([
            np.array(obj.position),
            np.array(obj.position) - back_vector * 0.01,
        ])

        # Variant B: origin at geometric center (most GLB assets)
        back_face_center = np.array(obj.position) - front_vector * (obj.dimensions[2] / 2)
        check_points.extend([
            back_face_center,
            back_face_center - back_vector * 0.01,
        ])
        
        # Also check multiple points across the back surface
        obj_half_width = obj.dimensions[0] / 2
        for point in check_points[:1]:  # Use first point as base
            check_points.extend([
                point + np.array([-obj_half_width, 0, 0]),  # Left edge
                point + np.array([obj_half_width, 0, 0]),   # Right edge
            ])
        
        for wall_mesh in wall_meshes:
            for check_point in check_points:
                # Method 1: Check distance from check point to wall surface
                closest_point, distance, _ = wall_mesh.nearest.on_surface([check_point])
                if len(distance) > 0 and distance[0] < self.config.support_tolerance:
                    if self.config.debug_output:
                        logger.debug("%s is attached to wall (distance: %.3fm)", obj.name, distance[0])
                    return True
                
                # Method 2: Raycast from check point towards the wall
                ray_origin = check_point - back_vector * 0.02  # Start 2cm behind
                hit, _, _ = wall_mesh.ray.intersects_location([ray_origin], [back_vector], multiple_hits=False)
                if len(hit) > 0:
                    hit_distance = np.linalg.norm(hit[0] - ray_origin)
                    if hit_distance < self.config.support_tolerance:
                        if self.config.debug_output:
                            logger.debug("%s is attached to wall (raycast hit: %.3fm)", obj.name, hit_distance)
                        return True
            
            # --- Fallback: bounding box proximity check (orientation-agnostic) ---
            try:
                bbox_vertices = self._object_meshes[obj.name].bounding_box_oriented.vertices
                if bbox_vertices.shape[0] > 24:  # sample at most 24 vertices for speed
                    bbox_vertices = bbox_vertices[:: max(1, bbox_vertices.shape[0] // 24)]
                _, distances, _ = wall_mesh.nearest.on_surface(bbox_vertices)
                if len(distances) > 0 and np.min(distances) < self.config.support_tolerance:
                    if self.config.debug_output:
                        logger.debug("%s is attached to wall via bbox proximity (min dist: %.3fm)", obj.name, float(np.min(distances)))
                    return True
            except Exception as e:
                if self.config.debug_output:
                    logger.debug("BBox proximity check failed for %s: %s", obj.name, e)
        
        if self.config.debug_output:
            logger.debug("%s is NOT attached to any wall", obj.name)
        return False

    def _check_ceiling_attachment_mesh(self, obj: SceneObject) -> bool:
        """Check ceiling attachment using upward raycast to ceiling mesh."""
        if obj.name not in self._object_meshes or self._ceiling_mesh is None:
            return True  # Cannot check, assume it's fine

        obj_mesh = self._object_meshes[obj.name]
        
        # Cast rays from the top surface of the object upwards
        # With bottom-centered positions, the top is at position_y + height
        top_center = np.array([obj.position[0], obj.position[1] + obj.dimensions[1], obj.position[2]])
        
        # Raycast from just above the object upwards
        ray_origin = top_center + np.array([0, 0.01, 0])
        ray_direction = np.array([0, 1, 0])
        
        # Check for hit within tolerance
        hit, _, _ = self._ceiling_mesh.ray.intersects_location([ray_origin], [ray_direction], multiple_hits=False)
        if len(hit) > 0:
            distance = np.linalg.norm(hit[0] - ray_origin)
            if distance < self.config.support_tolerance:
                return True
        
        logger.debug("%s is not attached to the ceiling", obj.name)
        return False

    def _fix_mesh_support(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Fix support for an object by finding the best support surface below it."""
        if obj.obj_type in [ObjectType.LARGE, ObjectType.SMALL]:
            return self._fix_surface_support_mesh(obj, context_objects)
        elif obj.obj_type == ObjectType.WALL:
            return self._fix_wall_support_mesh(obj)
        elif obj.obj_type == ObjectType.CEILING:
            return self._fix_ceiling_support_mesh(obj)
        return obj

    def _fix_surface_support_mesh(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Fix surface support by moving object to proper supported position."""
        if obj.name not in self._object_meshes:
            return obj

        obj_mesh = self._object_meshes[obj.name]
        
        # Position is bottom-centered, so the bottom Y is the position Y itself
        object_bottom_y: float = obj.position[1]

        support_tolerance: float = self.config.get_support_tolerance(obj.obj_type)

        max_support_y: float = -np.inf  # initialise to negative infinity so that we can detect *any* valid support
        hits_found: bool = False

        # --- Floor support ---------------------------------------------------
        if self._floor_mesh:
            ray_origin = np.array([obj.position[0], object_bottom_y + 1e-4, obj.position[2]])
            try:
                locations, _, _ = self._floor_mesh.ray.intersects_location([ray_origin], [[0, -1, 0]])
            except Exception:
                locations = []

            if len(locations) > 0:
                loc_arr = np.asarray(locations)
                floor_y: float = float(np.max(loc_arr[:, 1]))
                if floor_y <= object_bottom_y:  # Only count hits below current bottom
                    max_support_y = max(max_support_y, floor_y)
                    hits_found = True

        # --- Support from other objects -------------------------------------
        support_context = self._get_relevant_support_context(obj, context_objects)
        if self.config.debug_output:
            logger.debug("Support context: %s (from %d total)", [s.name for s in support_context], len(context_objects))

        for sup_obj in support_context:
            # First, try the precise ray-cast routine if a mesh is available.
            if sup_obj.name in self._object_meshes:
                supported, support_y = self._compute_support_from_object(
                    obj,
                    sup_obj,
                    object_bottom_y,
                    support_tolerance,
                )

                # get the highest support surface below the object
                if support_y is not None and support_y <= object_bottom_y:
                    max_support_y = max(max_support_y, support_y)
                    hits_found = True
                continue

            # use bounding box top face as fallback (position is bottom)
            sup_top_y: float = sup_obj.position[1] + sup_obj.dimensions[1]
            if sup_top_y <= object_bottom_y + support_tolerance:
                max_support_y = max(max_support_y, sup_top_y)
                hits_found = True

        if not hits_found:
            if self.config.debug_output:
                logger.debug("%s is NOT supported by any surface", obj.name)
                logger.debug("Checked %d support objects", len(support_context))
            # Nothing to do – leave object unchanged.
            return obj

        target_position_y: float = max_support_y + self.config.support_stability_offset

        current_y: float = float(obj.position[1])

        if abs(current_y - target_position_y) < self.config.support_position_threshold:
            # Adjustment smaller than threshold – keep current placement.
            return obj

        translation_y: float = target_position_y - current_y

        new_position = list(obj.position)
        new_position[1] = float(target_position_y) + 0.0001
        obj.position = (float(new_position[0]), float(new_position[1]), float(new_position[2]))

        # Update internal scene mesh position
        obj_mesh.apply_translation([0, translation_y, 0])

        if self.config.debug_output:
            logger.debug("Fixed surface support for %s: %.3fm -> %.3fm (Δ=%.3fm)", obj.name, current_y, target_position_y, translation_y)

        self.stats["support_fixes_applied"] += 1

        return obj

    def _fix_wall_support_mesh(self, obj: SceneObject) -> SceneObject:
        """Fix wall object attachment by moving it to the nearest wall."""
        wall_meshes = self._select_wall_meshes(obj)
        if obj.name not in self._object_meshes or not wall_meshes:
            return obj

        obj_mesh = self._object_meshes[obj.name]
        
        # Find the nearest wall and the point on it
        closest_point = None
        min_dist = float('inf')
        
        # Use object center to find the closest wall
        for wall_mesh in wall_meshes:
            point, dist, _ = wall_mesh.nearest.on_surface([obj_mesh.center_mass])
            if dist[0] < min_dist:
                min_dist = dist[0]
                closest_point = point[0]

        if closest_point is None:
            return obj

        # Get the "front" direction vector of the object
        angle_rad = np.radians(obj.rotation)
        front_vector = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
        back_vector = -front_vector

        # Add a small gap (0.5mm) to prevent z-fighting
        gap = 0.0005
        
        # Compute target position without assuming origin location
        # Try both hypotheses and pick the one requiring smaller translation

        offset_back_origin   = front_vector * gap  # back-origin variant
        offset_center_origin = front_vector * (obj.dimensions[2] / 2 + gap)  # centre-origin variant

        candidate_pos_back   = closest_point + offset_back_origin
        candidate_pos_center = closest_point + offset_center_origin

        # Select the candidate with the shorter displacement from the current
        # position (in the XZ-plane).  Y is preserved regardless.
        disp_back   = np.linalg.norm((candidate_pos_back - np.array(obj.position))[[0, 2]])
        disp_center = np.linalg.norm((candidate_pos_center - np.array(obj.position))[[0, 2]])

        # Select the candidate with the shorter displacement and preserve original height
        chosen_pos = candidate_pos_back if disp_back < disp_center else candidate_pos_center
        chosen_pos[1] = obj.position[1]  # Preserve original Y height

        new_position = chosen_pos
        
        # Move the object
        translation = np.array(new_position, dtype=float) - np.array(obj.position, dtype=float)
        obj.position = (float(new_position[0]), float(new_position[1]), float(new_position[2]))
        obj_mesh.apply_translation(translation)
        
        if self.config.debug_output:
            logger.debug("Fixed wall support for %s: moved to %s", obj.name, new_position)
        
        self.stats['support_fixes_applied'] += 1
        return obj

    def _fix_ceiling_support_mesh(self, obj: SceneObject) -> SceneObject:
        """Fix ceiling support by moving object to be flush with ceiling."""
        if obj.name not in self._object_meshes or self._ceiling_mesh is None:
            return obj

        obj_mesh = self._object_meshes[obj.name]
        
        # New bottom Y so that the top of the object touches the ceiling (small gap)
        ceiling_height = self.scene.room_height
        new_y = ceiling_height - obj.dimensions[1] - 0.01
        
        new_position = (obj.position[0], new_y, obj.position[2])
        
        # Move the object
        translation = np.array(new_position) - np.array(obj.position)
        obj.position = new_position
        obj_mesh.apply_translation(translation)
        
        if self.config.debug_output:
            logger.debug("Fixed ceiling support for %s, moved to Y=%.3f", obj.name, new_y)
        self.stats['support_fixes_applied'] += 1
        return obj

    def _update_object_mesh_position(self, obj: SceneObject) -> None:
        """Update the cached mesh position to match the object's new position."""
        if obj.name in self._object_meshes:
            # Reload mesh with new position
            new_mesh = self._load_single_object_mesh(obj)
            if new_mesh is not None:
                self._object_meshes[obj.name] = new_mesh

    def _update_object_mesh_position_efficient(self, obj: SceneObject) -> None:
        """Efficiently update the cached mesh position using transformation instead of reloading."""
        if obj.name not in self._object_meshes or obj.name not in self._base_object_meshes:
            return
        
        # Get the base mesh and current mesh
        base_mesh = self._base_object_meshes[obj.name]
        current_mesh = self._object_meshes[obj.name]
        
        # Calculate the required transformation using bottom-center alignment
        target_position = np.array(obj.position, dtype=float)
        # Current bottom center from AABB (fast and robust for our axis-aligned updates)
        current_bounds = current_mesh.bounds
        current_bottom_center = np.array([
            float((current_bounds[0][0] + current_bounds[1][0]) * 0.5),
            float(current_bounds[0][1]),
            float((current_bounds[0][2] + current_bounds[1][2]) * 0.5),
        ], dtype=float)
        
        # Apply rotation if needed
        if obj.rotation != 0:
            # Reset to base mesh and apply full transformation
            self._object_meshes[obj.name] = base_mesh.copy()
            mesh = self._object_meshes[obj.name]
            
            # Apply rotation
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=np.radians(obj.rotation),
                direction=[0, 1, 0],
                point=[0, 0, 0]
            )
            mesh.apply_transform(rotation_matrix)
            
            # Apply translation to match bottom-center
            translation_matrix = trimesh.transformations.translation_matrix(target_position)
            mesh.apply_transform(translation_matrix)
        else:
            # Simple translation update to align bottom centers
            translation = target_position - current_bottom_center
            if np.linalg.norm(translation) > 1e-9:  # Only update if significant change
                translation_matrix = trimesh.transformations.translation_matrix(translation)
                current_mesh.apply_transform(translation_matrix)

    def _get_relevant_collision_context(self, obj: SceneObject, all_context: List[SceneObject]) -> List[SceneObject]:
        """Get relevant objects for collision detection based on type-aware rules."""
        # Small objects: include parent, same-layer siblings, and other motifs
        # Large/wall/ceiling: collide with everything except children

        if obj.obj_type != ObjectType.SMALL:
            return [o for o in all_context if not self._is_parent_child_relationship(obj, o)]

        parent_name: str | None = getattr(obj, "parent_name", None)

        # Extract layer identifier from the motif-id (e.g. "layer_2") if present
        layer_tag: str | None = None
        if hasattr(obj, "motif_id") and obj.motif_id and "layer_" in obj.motif_id:
            try:
                layer_tag = obj.motif_id.split("layer_")[1].split("_")[0]  # yields "2" for "...layer_2_surface..."
            except Exception:
                layer_tag = None

        relevant: list[SceneObject] = []

        for other in all_context:
            # Always include the parent
            if parent_name and other.name == parent_name:
                relevant.append(other)
                continue

            # Skip objects with *different* parent to avoid cross-surface clashes
            if getattr(other, "parent_name", None) != parent_name:
                # object from a different motif – keep
                relevant.append(other)
                continue

            # Same parent – include only if on the same layer
            if layer_tag and hasattr(other, "motif_id") and other.motif_id and f"layer_{layer_tag}_" in other.motif_id:
                relevant.append(other)

        return relevant

    def _get_relevant_support_context(self, obj: SceneObject, all_context: List[SceneObject]) -> List[SceneObject]:
        """Get relevant objects for support validation based on type-aware rules."""
        # Small objects: only check support from parent
        # Other types: check support from all context objects

        if obj.obj_type != ObjectType.SMALL:
            if self.config.debug_output:
                logger.debug("Support context for %s object %s: all %d context objects", obj.obj_type.name, obj.name, len(all_context))
            return all_context

        parent_name: str | None = getattr(obj, "parent_name", None)

        if self.config.debug_output:
            logger.debug("SUPPORT CONTEXT DEBUG for small object %s:", obj.name)
            logger.debug("Parent name: '%s'", parent_name)
            logger.debug("Available context objects: %s", [o.name for o in all_context])
            logger.debug("Context object types: %s", [(o.name, o.obj_type.name) for o in all_context])

        if not parent_name:
            # No explicit parent – fall back to large objects (legacy behaviour)
            fallback_objects = [o for o in all_context if o.obj_type == ObjectType.LARGE]
            if self.config.debug_output:
                logger.debug("No parent specified, falling back to %d LARGE objects: %s", len(fallback_objects), [o.name for o in fallback_objects])
            return fallback_objects

        parent_objs = [o for o in all_context if o.name == parent_name]
        if self.config.debug_output:
            logger.debug("Found %d objects matching parent name '%s': %s", len(parent_objs), parent_name, [o.name for o in parent_objs])
            if not parent_objs:
                logger.debug("PARENT NOT FOUND! Available names: %s", [o.name for o in all_context])
                # Try case-insensitive match
                case_insensitive_matches = [o for o in all_context if o.name.lower() == parent_name.lower()]
                if case_insensitive_matches:
                    logger.debug("Found case-insensitive matches: %s", [o.name for o in case_insensitive_matches])
                    return case_insensitive_matches
                # Try partial match
                partial_matches = [o for o in all_context if parent_name.lower() in o.name.lower() or o.name.lower() in parent_name.lower()]
                if partial_matches:
                    logger.debug("Found partial matches: %s", [o.name for o in partial_matches])
                    return partial_matches
                logger.debug("No matches found, returning empty list")
        return parent_objs

    def _is_parent_child_relationship(self, obj1: SceneObject, obj2: SceneObject) -> bool:
        """Check if two objects have a parent-child relationship."""
        return ((hasattr(obj1, 'parent_name') and obj1.parent_name == obj2.name) or
                (hasattr(obj2, 'parent_name') and obj2.parent_name == obj1.name))

    def _apply_movement_constraints(self, direction: np.ndarray, obj_type: ObjectType) -> np.ndarray:
        """Apply movement constraints based on object type."""
        constrained_direction = direction.copy()
        
        if obj_type == ObjectType.LARGE:
            # LARGE objects stay on floor (no Y movement)
            constrained_direction[1] = 0.0
        elif obj_type == ObjectType.CEILING:
            # CEILING objects stay on ceiling (no Y movement)
            constrained_direction[1] = 0.0
        elif obj_type == ObjectType.WALL:
            # WALL objects can move along wall surface (limit movement away from wall)
            constrained_direction[1] *= self.config.wall_y_movement_factor
        # SMALL objects can move in all directions (no constraints)
        
        # Renormalize if needed
        norm = np.linalg.norm(constrained_direction)
        if norm > 1e-6:
            constrained_direction = constrained_direction / norm
        
        return constrained_direction

    def _group_objects_by_motif(self, objects: List[SceneObject]) -> Dict[str, List[SceneObject]]:
        """Group objects by their motif ID."""
        groups = {}
        for obj in objects:
            if hasattr(obj, 'motif_id') and obj.motif_id:
                if obj.motif_id not in groups:
                    groups[obj.motif_id] = []
                groups[obj.motif_id].append(obj)
        return groups

    def _get_objects_not_in_motifs(self, all_objects: List[SceneObject], motif_groups: Dict[str, List[SceneObject]]) -> List[SceneObject]:
        """Get objects that are not part of any motif."""
        motif_object_names = set()
        for objects in motif_groups.values():
            motif_object_names.update(obj.name for obj in objects)
        
        return [obj for obj in all_objects if obj.name not in motif_object_names]

    def _print_summary(self):
        """Print optimization summary."""
        if self.config.debug_output:
            logger.info("="*60)
            logger.info("Unified Mesh-Based Optimization Summary")
            logger.info("="*60)
            logger.info("Objects processed: %d", self.stats['objects_processed'])
            logger.info("Collisions resolved: %d", self.stats['collisions_resolved'])
            logger.info("Support fixes applied: %d", self.stats['support_fixes_applied'])
            logger.info("Early exits: %d", self.stats['early_exits'])
            logger.info("Processing time: %.2fs", self.stats['processing_time'])
            logger.info("="*60)

    def _has_hssd_transform(self, obj: SceneObject) -> bool:
        """Check if object has HSSD transform and needs special origin handling."""
        if hasattr(obj, 'has_hssd_alignment') and callable(obj.has_hssd_alignment):
            try:
                # `has_hssd_alignment` already checks for a valid matrix after our
                # recent patch – if it returns *True* we can safely assume the
                # transform exists.
                if obj.has_hssd_alignment():
                    return True
            except Exception:
                pass

        # Fallback: Inspect preprocessing data directly (legacy objects or tests)
        if hasattr(obj, '_preprocessing_data') and obj._preprocessing_data:
            transforms = obj._preprocessing_data.get('transforms', {})
            for transform_type, tinfo in transforms.items():
                if 'hssd' in transform_type.lower() and getattr(tinfo, 'transform_matrix', None) is not None:
                    return True

        return False

    def _calculate_penetration_depth(self, obj: SceneObject, colliding_objects: List[SceneObject]) -> float:
        """Calculate the approximate penetration depth using existing collision detection."""
        if not colliding_objects or obj.name not in self._object_meshes:
            return 0.0
        
        obj_mesh = self._object_meshes[obj.name]
        max_penetration = 0.0
        
        for other_obj in colliding_objects:
            if other_obj.name in self._object_meshes:
                other_mesh = self._object_meshes[other_obj.name]
                
                # Reuse the collision detection logic with penetration depth
                collision_result = self._check_mesh_collision(
                    obj_mesh, other_mesh, return_penetration=True
                )
                
                if isinstance(collision_result, tuple):
                    is_collision, penetration = collision_result
                else:
                    is_collision = collision_result
                    penetration = 0.0
                
                if is_collision:
                    max_penetration = max(max_penetration, penetration)
        
        return max_penetration

    def _select_wall_meshes(self, obj: SceneObject) -> List[trimesh.Trimesh]:
        """Return wall meshes relevant for the given object.

        If the object has a `wall_id` attribute we try to find the corresponding
        mesh inside the room geometry. This prevents the optimizer from
        mistakenly snapping the object to an unintended wall when the support
        validation fails for numerical reasons.

        Args:
            obj: Scene object under consideration.

        Returns:
            A list with the wall mesh(es) to check against. If the requested
            wall mesh cannot be found we gracefully fall back to *all* wall
            meshes so the original behaviour is preserved.
        """
        # If no specific wall is requested keep legacy behaviour
        if not getattr(obj, "wall_id", None):
            return self._wall_meshes

        # Try to look the mesh up in the loaded room scene
        if self._room_scene and obj.wall_id in self._room_scene.geometry:
            return [self._room_scene.geometry[obj.wall_id]]

        # Fallback: we could not find the requested wall mesh, keep legacy behaviour
        return self._wall_meshes

    def _get_wall_object_support_height(self, wall_obj: SceneObject) -> float:
        """Get the Y position of the top surface of a wall object for placing small objects."""
        has_hssd = self._has_hssd_transform(wall_obj)
        
        if has_hssd:
            # For HSSD wall objects, the origin is typically at the back-bottom face
            # The top surface for placing objects is at: origin_y + height
            support_y = wall_obj.position[1] + wall_obj.dimensions[1]
        else:
            # Standard wall objects use bottom-centered positioning as well
            support_y = wall_obj.position[1] + wall_obj.dimensions[1]
        
        if self.config.debug_output:
            logger.debug("Wall object %s support height: %.3fm (%s positioning)", wall_obj.name, support_y, 'HSSD' if has_hssd else 'standard')
        
        return support_y

    def _check_wall_object_support(self, small_obj: SceneObject, wall_obj: SceneObject, support_tolerance: float) -> bool:
        """Check if a small object is geometrically positioned on a wall object."""
        # Check if wall object has HSSD transform which affects origin position
        has_hssd = self._has_hssd_transform(wall_obj)
        
        # Get the correct wall top surface height
        wall_top_y = self._get_wall_object_support_height(wall_obj)
        
        if has_hssd:
            # For horizontal extents, HSSD objects still use center-based positioning
            wall_x_min = wall_obj.position[0] - wall_obj.dimensions[0] / 2
            wall_x_max = wall_obj.position[0] + wall_obj.dimensions[0] / 2

            # For Z (depth), HSSD wall objects have origin at back, so front surface is at origin + depth
            wall_z_min = wall_obj.position[2]
            wall_z_max = wall_obj.position[2] + wall_obj.dimensions[2]

            if self.config.debug_output:
                logger.debug("HSSD wall object detected - using back-face origin positioning")
                logger.debug("Wall Z range: [%.3f, %.3f] (origin + depth)", wall_z_min, wall_z_max)
        else:
            # Standard wall object with center-based origin
            wall_x_min = wall_obj.position[0] - wall_obj.dimensions[0] / 2
            wall_x_max = wall_obj.position[0] + wall_obj.dimensions[0] / 2
            wall_z_min = wall_obj.position[2] - wall_obj.dimensions[2] / 2
            wall_z_max = wall_obj.position[2] + wall_obj.dimensions[2] / 2
        
        # Small object bottom (position is bottom-centered)
        small_obj_bottom_y = small_obj.position[1]
        
        # Check vertical alignment - small object bottom should be near wall object top
        vertical_distance = abs(small_obj_bottom_y - wall_top_y)
        if vertical_distance > support_tolerance:
            if self.config.debug_output:
                logger.debug("Wall support check: vertical distance %.3fm > tolerance %.3fm", vertical_distance, support_tolerance)
                logger.debug("Wall top Y: %.3fm", wall_top_y)
                logger.debug("Small bottom Y: %.3fm", small_obj_bottom_y)
            return False
        
        # Check horizontal overlap
        small_x_min = small_obj.position[0] - small_obj.dimensions[0] / 2
        small_x_max = small_obj.position[0] + small_obj.dimensions[0] / 2
        small_z_min = small_obj.position[2] - small_obj.dimensions[2] / 2
        small_z_max = small_obj.position[2] + small_obj.dimensions[2] / 2
        
        x_overlap = not (small_x_max < wall_x_min or small_x_min > wall_x_max)
        z_overlap = not (small_z_max < wall_z_min or small_z_min > wall_z_max)
        
        if self.config.debug_output:
            logger.debug("Wall support check for %s on %s (%s):", small_obj.name, wall_obj.name, 'HSSD' if has_hssd else 'standard')
            logger.debug("Vertical distance: %.3fm (tolerance: %.3fm)", vertical_distance, support_tolerance)
            logger.debug("X overlap: %s (small: [%.2f, %.2f], wall: [%.2f, %.2f])", x_overlap, small_x_min, small_x_max, wall_x_min, wall_x_max)
            logger.debug("Z overlap: %s (small: [%.2f, %.2f], wall: [%.2f, %.2f])", z_overlap, small_z_min, small_z_max, wall_z_min, wall_z_max)
        
        return x_overlap and z_overlap

    def _resolve_single_mesh_collision_mtv(self, obj: SceneObject, colliding_objects: List[SceneObject]) -> SceneObject:
        """Resolve collisions using the Minimum-Translation Vector.

        For every colliding pair we query the contact manifold.  We
        construct the MTV as ``normal * (depth + ε)`` where *ε* is a small
        clearance.  If multiple contacts are present we pick the one with the
        largest penetration depth that guarantees separation in a single
        step while keeping displacement minimal.

        The resulting MTV is then constrained according to object-type rules
        (floor-bound, ceiling-bound, …) and applied once.
        """

        resolved_obj = copy.deepcopy(obj)
        if not colliding_objects:
            return resolved_obj

        if obj.name not in self._object_meshes:
            # Cannot resolve without our own mesh – keep original behaviour
            return resolved_obj

        obj_mesh = self._object_meshes[obj.name]

        epsilon = 1e-4

        best_mtv: Optional[np.ndarray] = None
        best_penetration: float = 0.0

        for other in colliding_objects:
            if other.name not in self._object_meshes:
                continue

            other_mesh = self._object_meshes[other.name]

            # Build a tiny collision manager for the pair to get contact data
            cm = trimesh.collision.CollisionManager()
            cm.add_object("obj", obj_mesh)
            cm.add_object("other", other_mesh)
            result = cm.in_collision_internal(return_data=True)
            # Trimesh may return a tuple of (bool, contacts) or (bool, set, list)
            in_collision = False
            contacts = None
            if isinstance(result, tuple):
                if len(result) >= 1:
                    in_collision = bool(result[0])
                if len(result) >= 2:
                    contacts = result[1]
            else:
                in_collision = bool(result)
                contacts = None
            if not in_collision or not contacts:
                continue

            # Choose deepest contact among the list returned
            deepest_contact = max(contacts, key=lambda c: float(c.depth))
            penetration: float = float(deepest_contact.depth)
            normal_vec = np.array(deepest_contact.normal, dtype=float)

            # Ensure the normal points *away* from the other object – if the dot
            # product between the normal and centre-to-centre vector is
            # negative we flip the normal.
            center_delta = np.array(obj.position) - np.array(other.position)
            if np.dot(normal_vec, center_delta) < 0:
                normal_vec = -normal_vec

            mtv = normal_vec * (penetration + epsilon)

            if penetration > best_penetration:
                best_penetration = penetration
                best_mtv = mtv

        if best_mtv is None:
            # Could not compute an MTV (e.g. no contact normals) – fallback
            return resolved_obj

        mtv_len: float = float(np.linalg.norm(best_mtv))
        if mtv_len < 1e-9:
            return resolved_obj

        # Constrain movement along allowed axes based on object type
        direction_unit = best_mtv / mtv_len
        constrained_dir = self._apply_movement_constraints(direction_unit, resolved_obj.obj_type)
        best_mtv = constrained_dir * mtv_len

        # Apply translation
        new_position = np.array(resolved_obj.position, dtype=float) + np.array(best_mtv, dtype=float)
        resolved_obj.position = (float(new_position[0]), float(new_position[1]), float(new_position[2]))

        return resolved_obj

    def _fix_wall_object_support(self, small_obj: SceneObject, wall_obj: SceneObject) -> SceneObject:
        """Fix positioning of a small object to sit properly on a wall object surface."""
        # Get the correct wall support height
        wall_support_y = self._get_wall_object_support_height(wall_obj)
        
        # Calculate the target bottom Y position for the small object
        target_bottom_y = wall_support_y + self.config.support_stability_offset
        
        # Calculate translation needed
        current_y = small_obj.position[1]
        translation_y = target_bottom_y - current_y
        
        # Apply the translation
        new_position = list(small_obj.position)
        new_position[1] = float(target_bottom_y)
        small_obj.position = (float(new_position[0]), float(new_position[1]), float(new_position[2]))
        
        # Update mesh position if available
        if small_obj.name in self._object_meshes:
            obj_mesh = self._object_meshes[small_obj.name]
            obj_mesh.apply_translation([0, translation_y, 0])
        
        if self.config.debug_output:
            logger.debug("Fixed wall object support for %s: %.3fm -> %.3fm (Δ=%.3fm)",
                       small_obj.name, current_y, target_bottom_y, translation_y)
            logger.debug("Wall support surface at %.3fm, object bottom now at %.3fm", wall_support_y, target_bottom_y)
        
        self.stats['support_fixes_applied'] += 1
        return small_obj

    def _compute_support_from_object(
        self,
        obj: SceneObject,
        sup_obj: SceneObject,
        object_bottom_y: float,
        support_tolerance: float,
    ) -> Tuple[bool, Optional[float]]:
        """Return whether *sup_obj* provides vertical support for *obj* and the
        corresponding support surface height.

        This routine centralises the ray-casting logic that was previously
        duplicated in :py:meth:`_check_surface_support_mesh` and
        :py:meth:`_fix_surface_support_mesh`.

        Args:
            obj: The object that needs support.
            sup_obj: Candidate support object.
            object_bottom_y: The *world-space* y-coordinate of the bottom face of
                *obj* (centre y − half height).
            support_tolerance: Maximum allowed gap between *obj* and the support
                surface for the configuration to be considered valid.

        Returns:
            A tuple ``(is_supported, support_y)`` where ``is_supported`` is
            ``True`` if the candidate provides adequate support according to the
            tolerance and ``support_y`` is the *y* value of the support surface
            (``None`` when no valid support has been found).
        """

        # Bail out early if we do not have a mesh for the support object.
        if sup_obj.name not in self._object_meshes:
            return False, None

        sup_mesh = self._object_meshes[sup_obj.name]

        # --- Build ray-cast sample points (centre + four corners) ---
        check_points: list[np.ndarray] = [
            np.array([obj.position[0], object_bottom_y + 1e-5, obj.position[2]])
        ]

        half_width: float = float(obj.dimensions[0] * 0.5 * 0.95)  # stay inside
        half_depth: float = float(obj.dimensions[2] * 0.5 * 0.95)

        for dx, dz in [
            (half_width, half_depth),
            (-half_width, half_depth),
            (half_width, -half_depth),
            (-half_width, -half_depth),
        ]:
            check_points.append(
                np.array([obj.position[0] + dx, object_bottom_y + 1e-5, obj.position[2] + dz])
            )

        hit_any: bool = False
        support_y: Optional[float] = None

        for ray_origin in check_points:
            ray_direction = np.array([0, -1, 0])
            try:
                hits, _, _ = sup_mesh.ray.intersects_location([ray_origin], [ray_direction])
            except Exception:
                # Robustness: occasionally the ray-casting backend raises for
                # degenerate triangles – treat as "no hit".
                hits = []

            if len(hits) == 0:
                continue

            # Use closest intersection (sorted by distance).
            hits_arr = np.asarray(hits)
            candidate_y: float = float(hits_arr[0, 1])
            distance: float = abs(candidate_y - ray_origin[1])

            # Record the highest intersection below the object
            if candidate_y <= object_bottom_y:
                support_y = candidate_y if support_y is None else max(support_y, candidate_y)

            # Mark as "supported" only when the gap satisfies the tolerance.
            if distance < support_tolerance:
                hit_any = True

        # Special case: small object on wall object - use geometric support when raycasting inconclusive
        if (
            not hit_any
            and sup_obj.obj_type == ObjectType.WALL
            and obj.obj_type == ObjectType.SMALL
            and self._check_wall_object_support(obj, sup_obj, support_tolerance)
        ):
            support_y_geom = self._get_wall_object_support_height(sup_obj)
            return True, support_y_geom

        return hit_any, support_y