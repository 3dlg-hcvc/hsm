import time
import trimesh
import numpy as np
from copy import deepcopy
import logging

from .spatial_optimizer import optimize as optimize_single_arrangement

logger = logging.getLogger(__name__)


def optimize_hierarchical(
    hierarchy,
    resolve_collisions: bool = True,
    collision_move_step: float = 0.005,
    collision_max_iters: int = 1000,
    make_tight: bool = False,
    make_tight_iters: int = 10,
    approximate_gravity: bool = True,
):
    """Optimize arrangements hierarchically"""
    
    if not hierarchy.root:
        logger.warning("Empty hierarchy provided to hierarchical optimizer")
        return hierarchy
    
    logger.info(f"Hierarchical spatial optimization starting...")
    overall_start_time = time.time()
    
    # Phase 1: Optimize all leaf nodes (including root if it's a leaf)
    leaf_nodes = hierarchy.get_leaf_nodes()
    logger.info(f"Phase 1: Optimizing {len(leaf_nodes)} leaf nodes")
    
    for leaf_node in leaf_nodes:
        if leaf_node.arrangement and leaf_node.arrangement.objs:
            logger.info(f"  Optimizing leaf node: {leaf_node.motif_type} (depth={leaf_node.depth}) with {len(leaf_node.arrangement.objs)} objects")
            node_start_time = time.time()
            
            optimized_arrangement = optimize_single_arrangement(
                leaf_node.arrangement,
                resolve_collisions=resolve_collisions,
                collision_move_step=collision_move_step,
                collision_max_iters=collision_max_iters,
                make_tight=make_tight,
                make_tight_iters=make_tight_iters,
                approximate_gravity=approximate_gravity
            )
            
            hierarchy.set_arrangement(leaf_node, optimized_arrangement)
            logger.info(f"    Leaf optimization completed in {(time.time() - node_start_time):.3f}s")
        else:
            logger.info(f"  Leaf node '{leaf_node.motif_type}' at depth {leaf_node.depth} - no arrangement to optimize")
    
    # Phase 2: Optimize only non-leaf nodes (parent arrangements that contain sub-arrangements)
    logger.info("Phase 2: Optimizing non-leaf arrangements (parent motifs)")
    non_leaf_nodes = [node for node in hierarchy.traverse_bottom_up() if node.children]
    
    if not non_leaf_nodes:
        logger.info("  No non-leaf nodes to optimize (simple arrangement)")
    
    for node in non_leaf_nodes:
        if node.arrangement and node.arrangement.objs:
            logger.info(f"  Optimizing parent motif: {node.motif_type} (depth={node.depth})")
            node_start_time = time.time()
            
            optimized_arrangement = optimize_single_arrangement(
                node.arrangement,
                resolve_collisions=resolve_collisions,
                collision_move_step=collision_move_step,
                collision_max_iters=collision_max_iters,
                make_tight=make_tight,
                make_tight_iters=make_tight_iters,
                approximate_gravity=approximate_gravity
            )
            
            hierarchy.set_arrangement(node, optimized_arrangement)
            logger.info(f"    Completed in {(time.time() - node_start_time):.3f}s")
    
    # Phase 3: Optimize motif-to-motif relationships at each depth level
    logger.info("Phase 3: Inter-motif optimization (by depth level)")
    for depth in range(hierarchy.root.depth, max(node.depth for node in hierarchy.execution_order) + 1):
        nodes_at_depth = hierarchy.get_nodes_at_depth(depth)
        if len(nodes_at_depth) <= 1:
            continue
            
        logger.info(f"  Optimizing {len(nodes_at_depth)} motifs at depth {depth}")
        depth_start_time = time.time()
        
        combined_arrangement = _create_combined_arrangement(nodes_at_depth)
        if combined_arrangement and len(combined_arrangement.objs) > 1:
            optimized_combined = optimize_single_arrangement(
                combined_arrangement,
                resolve_collisions=resolve_collisions,
                collision_move_step=collision_move_step * 2,
                collision_max_iters=collision_max_iters // 2,
                make_tight=False,
                approximate_gravity=approximate_gravity
            )

            _distribute_transforms_to_motifs(nodes_at_depth, combined_arrangement, optimized_combined)
        
        logger.info(f"    Completed depth {depth} in {(time.time() - depth_start_time):.3f}s")
    
    logger.info(f"Hierarchical optimization completed in {(time.time() - overall_start_time):.3f}s")
    return hierarchy


def _create_combined_arrangement(nodes):
    """Create a combined arrangement from multiple motif nodes."""
    from hsm_core.scene_motif import Arrangement, HierarchyNode
    if not nodes:
        return None
    
    combined_objs = []
    
    for i, node in enumerate(nodes):
        if not node.arrangement or not node.arrangement.objs:
            continue
            
        # Create a combined mesh for this motif
        motif_meshes = []
        for obj in node.arrangement.objs:
            if hasattr(obj, 'mesh') and obj.mesh is not None:
                obj_mesh = deepcopy(obj.mesh)
                if hasattr(obj, 'bounding_box') and obj.bounding_box:
                    obj_mesh.apply_transform(obj.bounding_box.no_scale_matrix)
                motif_meshes.append(obj_mesh)
                
        # Combine all meshes in this motif into a single mesh
        if motif_meshes:
            if len(motif_meshes) == 1:
                combined_mesh = motif_meshes[0]
            else:
                combined_mesh = trimesh.util.concatenate(motif_meshes)
            
            # Create a super-object representing this entire motif
            from hsm_core.scene_motif.core.obj import Obj
            from hsm_core.scene_motif.core.bounding_box import BoundingBox
            
            bounds = combined_mesh.bounds
            centroid = combined_mesh.centroid
            half_size = (bounds[1] - bounds[0]) / 2
            
            super_obj = Obj(
                label=f"motif_{node.motif_type}_{i}",
                mesh=combined_mesh,
                bounding_box=BoundingBox(
                    centroid=centroid,
                    half_size=half_size,
                    coord_axes=np.eye(3)
                )
            )
            
            setattr(super_obj, '_motif_node', node)
            combined_objs.append(super_obj)
    
    if not combined_objs:
        return None
    
    return Arrangement(
        combined_objs,
        f"Combined arrangement of {len(nodes)} motifs",
        "combined_motifs()"
    )


def _distribute_transforms_to_motifs(
    nodes,
    original_combined,
    optimized_combined
) -> None:
    """Distribute the transforms from optimized combined arrangement back to individual motifs."""
    if len(original_combined.objs) != len(optimized_combined.objs):
        logger.warning("Mismatch in object count between original and optimized combined arrangements")
        return
    
    for orig_obj, opt_obj in zip(original_combined.objs, optimized_combined.objs):
        if not hasattr(orig_obj, '_motif_node'):
            continue
            
        motif_node = orig_obj._motif_node
        
        orig_centroid = orig_obj.bounding_box.centroid
        opt_centroid = opt_obj.bounding_box.centroid
        translation = opt_centroid - orig_centroid
        
        if motif_node.arrangement:
            for obj in motif_node.arrangement.objs:
                if hasattr(obj, 'bounding_box') and obj.bounding_box:
                    obj.bounding_box.centroid += translation
    
    logger.debug(f"Distributed transforms to {len(nodes)} motifs")


def optimize_with_hierarchy(
    arrangement,
    hierarchy=None,
    **kwargs
):
    """Optimize an arrangement with optional hierarchy support."""

    if hierarchy and hierarchy.root:
        optimized_hierarchy = optimize_hierarchical(hierarchy, **kwargs)
        
        if optimized_hierarchy.root and optimized_hierarchy.root.arrangement:
            return optimized_hierarchy.root.arrangement
        else:
            logger.warning("Hierarchical optimization failed, falling back to standard optimization")
            return optimize_single_arrangement(arrangement, **kwargs)
    else:
        return optimize_single_arrangement(arrangement, **kwargs) 