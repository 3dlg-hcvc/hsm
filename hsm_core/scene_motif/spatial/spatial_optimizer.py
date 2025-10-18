import time
import trimesh
import trimesh.sample
import numpy as np
from copy import deepcopy
from hsm_core.utils import get_logger
from hsm_core.scene_motif.core.arrangement import Arrangement

logger = get_logger('scene_motif.spatial.spatial_optimizer')


def optimize(
    arrangement: Arrangement,
    resolve_collisions: bool = True,
    collision_move_step: float = 0.005,
    collision_max_iters: int = 1000,
    make_tight: bool = False,
    make_tight_iters: int = 10,
    approximate_gravity: bool = True,
) -> Arrangement:
    '''
    Optimize the spatial positions and orientations of the objects in the arrangement 
    such that they are physically possible.

    Args:
        arrangement: Arrangement, the arrangement to optimize
        resolve_collisions: bool, whether to resolve collisions between objects
        collision_move_step: float, the distance per step to move the objects when resolving collisions
        collision_max_iters: int, the maximum number of iterations to resolve collisions
        make_tight: bool, whether to make the objects fit tightly together
        make_tight_iters: int, the number of iterations to make the objects fit tightly together
        approximate_gravity: bool, whether to approximate gravity

    Returns:
        optimized_arrangement: Arrangement, the optimized arrangement
    '''

    # Pre-filter objects with meshes once
    obj_with_mesh = [obj for obj in arrangement.objs if obj.has_mesh]
    
    # If no objects have meshes, return the original arrangement
    if not obj_with_mesh:
        logger.warning("No objects with meshes found in arrangement")
        return arrangement  # Return original instead of None

    all_meshes = deepcopy([obj.mesh for obj in obj_with_mesh])

    # Initialize managers and timing
    collision_manager = trimesh.collision.CollisionManager()
    scene = trimesh.Scene()
    current_static_union = None

    # Include function_call label for hierarchical context
    label = arrangement.function_call or ""
    short_label = label.split('(')[0] if label and '(' in label else label
    logger.info(f"Spatial optimization [{short_label}] with {len(all_meshes)} meshes (resolve_collisions: {resolve_collisions}, make_tight: {make_tight}, approximate_gravity: {approximate_gravity})")
    overall_start_time = time.time()

    # Initialize the applied transformations for each object to the identity matrix
    # All transformations used during optimization are stored in this array, which is then used to update the arrangement at the end
    applied_transformations = np.tile(np.eye(4), (len(obj_with_mesh), 1, 1))
    
    for i, (obj, mesh) in enumerate(zip(obj_with_mesh, all_meshes)):
        mesh.apply_transform(obj.bounding_box.no_scale_matrix)
        
        # Skip the first mesh as there is nothing to compare against
        if i >= 1 and isinstance(arrangement, Arrangement):
        
            # ------------------------------------------------------------------------------ Pull objects together for a tight fit

            # Make the object fit tightly with the previous objects by moving it towards the centroid of the previous object
            if make_tight:
                combined_static_mesh = trimesh.util.concatenate(all_meshes[:i])
                
                for tight_iter in range(make_tight_iters):
                    
                    # Find the direction to move the object towards the centroid of the previous object
                    centroid_direction = all_meshes[i - 1].centroid - mesh.centroid
                    norm = np.linalg.norm(centroid_direction)
                    if norm < 1e-6:
                        logger.warning(f"Skipping tight fit for mesh {i} because its centroid overlaps with a previous mesh.")
                        break
                    centroid_direction /= norm

                    # Get the visible points of the object facing the centroid direction
                    surface_pts, face_idxs = trimesh.sample.sample_surface_even(mesh, 2048)
                    normals = mesh.face_normals[face_idxs]
                    visible_pts = surface_pts[np.dot(normals, centroid_direction) > 0.0]

                    # Check if any points are facing the right direction
                    if visible_pts.shape[0] == 0:
                        logger.warning(f"Make tight for mesh {i} could not find any visible points facing the target. Stopping.")
                        break
                    
                    # Find the intersection points of the rays from the visible points towards the centroid direction
                    ray_origins = visible_pts
                    ray_directions = np.tile(centroid_direction, (len(visible_pts), 1))
                    ray_intersections_pts, ray_idxs, _ = combined_static_mesh.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)

                    # Move the object towards the centroid direction until it touches the other objects
                    if len(ray_intersections_pts) > 0:
                        corresponding_ray_origins = ray_origins[ray_idxs]
                        # Find the minimum distance to move the object
                        distances = np.linalg.norm(corresponding_ray_origins - ray_intersections_pts, axis=1)

                        # Move the object by the minimum distance, weighted by the iteration
                        move_distance = np.min(distances) * (0.5 + 0.5 * (tight_iter+1) / make_tight_iters)

                        # Move the object
                        translation = centroid_direction * move_distance
                        translation_matrix = trimesh.transformations.translation_matrix(translation)
                        mesh.apply_transform(translation_matrix)
                        applied_transformations[i] = np.dot(translation_matrix, applied_transformations[i])
                        logger.info(f"Make tight for mesh {i} (iter {tight_iter+1}): moved object by {move_distance:.4f}m.")
                    else:
                        # Add logging for the silent failure case
                        logger.warning(f"Make tight for mesh {i} failed: Ray intersection found no points. Stopping tightening.")
                        break
        
            # ------------------------------------------------------------------------------ Push objects apart if in collision
            # Resolve collisions by moving the object away from the other objects
            if resolve_collisions:
                for collision_iter in range(collision_max_iters):
                    in_collision, contacts = collision_manager.in_collision_single(mesh, return_data=True)
                    if not in_collision:
                        break

                    # Find the direction to separate the objects
                    contact_pts = np.array([contact.point for contact in contacts])
                    separate_direction = np.mean(mesh.centroid - contact_pts, axis=0)

                    # Weight the direction based on the size of the object
                    extents = mesh.bounding_box_oriented.extents
                    safe_extents = np.where(extents == 0, 1e-6, extents)
                    weights = 1 / safe_extents
                    weighted_direction = separate_direction * weights
                    norm = np.linalg.norm(weighted_direction)
                    if norm > 1e-6:
                        weighted_direction /= norm
                    else:
                        weighted_direction = np.array([1.0, 0.0, 0.0])

                    # Adaptive step size
                    step = collision_move_step # Default step
                    if current_static_union:
                        mb_min, mb_max = mesh.bounds
                        sb_min, sb_max = current_static_union.bounds
                        overlap = np.minimum(sb_max - mb_min, mb_max - sb_min)
                        positive = overlap[overlap > 0]
                        raw_step = positive.min() if len(positive) > 0 else collision_move_step
                        step = max(min(raw_step, collision_move_step), collision_move_step * 0.1)

                    # Move the object in the direction of the weighted direction
                    translation = weighted_direction * step
                    translation_matrix = trimesh.transformations.translation_matrix(translation)
                    mesh.apply_transform(translation_matrix)
                    applied_transformations[i] = np.dot(translation_matrix, applied_transformations[i])

        # Add the optimized mesh to the collision manager for the next iteration
        collision_manager.add_object(f"mesh_{i}", mesh)
        scene.add_geometry(mesh, node_name=f"mesh_{i}") # Ensure node name is consistent
        
        # Update current_static_union for the next iteration
        if current_static_union is None:
            current_static_union = mesh.copy()
        else:
            current_static_union = trimesh.util.concatenate([current_static_union, mesh.copy()])

    # ------------------------------------------------------------------------------ Approximate gravity

    if approximate_gravity:       
        settled_meshes_union = None
         
        # Sort meshes by their centroid Y position to process them from bottom to top.
        indexed_meshes = list(enumerate(all_meshes))        
        sorted_indexed_meshes = sorted(indexed_meshes, key=lambda item: item[1].centroid[1])
        
        for i, (orig_idx, mesh) in enumerate(sorted_indexed_meshes):
            logger.debug(f"  {i+1}. Object {orig_idx} at Y centroid: {mesh.centroid[1]:.3f}")

        # Create a floor plane below all objects to act as the ultimate ground.
        combined_static_mesh: trimesh.Trimesh = trimesh.util.concatenate(all_meshes)
        global_min_y = np.min(combined_static_mesh.bounds[:, 1])
        floor_plane: trimesh.Trimesh = trimesh.creation.box(extents=[10, 0.01, 10], transform=trimesh.transformations.translation_matrix([0, global_min_y - 0.005, 0]))

        # Process each mesh in bottom-up order.
        for process_idx, (original_index, mesh) in enumerate(sorted_indexed_meshes):
            logger.debug(f"Processing object {original_index} (step {process_idx + 1}/{len(sorted_indexed_meshes)})")
            
            # Sample points on the bottom-facing surfaces of the current mesh.
            surface_pts, face_idxs = trimesh.sample.sample_surface_even(mesh, 2048)
            if len(surface_pts) == 0:
                logger.debug(f"No surface points for object {original_index}, skipping")
                continue

            normals = mesh.face_normals[face_idxs]
            # Consider points where the normal has a negative Y component (facing down).
            ground_facing_mask = normals[:, 1] < -0.1  # More lenient threshold for downward-facing
            ground_facing_pts = surface_pts[ground_facing_mask]
            
            if len(ground_facing_pts) == 0:
                logger.debug(f"No downward-facing points for object {original_index}, skipping")
                continue

            ray_origins = ground_facing_pts
            ray_directions = np.tile([0, -1, 0], (len(ground_facing_pts), 1))

            # The potential support structure is the floor plus any meshes already settled.
            support_structure = floor_plane
            if settled_meshes_union is not None:
                support_structure = trimesh.util.concatenate([settled_meshes_union, floor_plane])

            # Find where the downward rays hit the support structure.
            intersection_pts, ray_idxs, _ = support_structure.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)

            if len(intersection_pts) > 0:
                # Calculate the vertical distance to the nearest support point below.
                corresponding_ray_origins = ray_origins[ray_idxs]
                distances = corresponding_ray_origins[:, 1] - intersection_pts[:, 1]
                
                # Only apply gravity if there's a significant gap (avoid tiny adjustments)
                min_distance_to_drop = np.min(distances)
                if min_distance_to_drop > 0.001:  # 1mm threshold
                    logger.debug(f"Dropping object {original_index} by {min_distance_to_drop:.3f}m")
                    
                    # Apply the gravity translation.
                    gravity_translation = [0, -min_distance_to_drop, 0]
                    translation_matrix = trimesh.transformations.translation_matrix(gravity_translation)
                    mesh.apply_transform(translation_matrix)
                    # Update the transformation for this specific object using its original index.
                    applied_transformations[original_index] = np.dot(translation_matrix, applied_transformations[original_index])
                else:
                    logger.debug(f"Object {original_index} already properly supported (gap: {min_distance_to_drop:.4f}m)")
            else:
                logger.debug(f"No support intersection found for object {original_index}")

            # Add the now-settled mesh to the union for subsequent objects to rest on.
            if settled_meshes_union is None:
                settled_meshes_union = mesh.copy()
            else:
                settled_meshes_union = trimesh.util.concatenate([settled_meshes_union, mesh.copy()])

        # Remove the floor plane from the scene
        try:
            if "floor_plane" in scene.geometry:
                scene.delete_geometry("floor_plane")
        except Exception as e:
            pass
        
    # ------------------------------------------------------------------------------ Update the arrangement with the applied transformations
    opt_objs = deepcopy(arrangement.objs)
    optimized_arrangement = Arrangement(
        objs=opt_objs,
        description=arrangement.description 
            if hasattr(arrangement, 'description') 
            else "Optimized arrangement",
        function_call=arrangement.function_call
    )
    optimized_arrangement.glb_path = arrangement.glb_path
    
    # map the original object to the optimized object
    obj_with_mesh_indices = { id(obj): i for i, obj in enumerate(obj_with_mesh) }
    for orig_obj, new_obj  in zip(arrangement.objs, opt_objs):
        if not orig_obj.has_mesh:
            continue
        mesh_idx = obj_with_mesh_indices[id(orig_obj)]
        tf = applied_transformations[mesh_idx]
        
        # homogeneous multiply
        c4 = np.append(new_obj.bounding_box.centroid, 1.0)
        new_obj.bounding_box.centroid = (tf @ c4)[:3]
    
    logger.info(f"[{short_label}] Optimized all objects in {(time.time() - overall_start_time):.3f} s")
    return optimized_arrangement
