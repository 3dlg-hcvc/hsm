import time
import trimesh
import numpy as np
from copy import deepcopy

from .spatial_optimizer import optimize as optimize_single_arrangement

import logging
logger = logging.getLogger('scene_motif.spatial')


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
    
    # Phase 2: Optimize only non-leaf nodes (parent motifs) by respecting children as super-objects
    logger.info("Phase 2: Optimizing non-leaf arrangements (parent motifs)")
    non_leaf_nodes = [node for node in hierarchy.traverse_bottom_up() if node.children]
    
    if not non_leaf_nodes:
        logger.info("  No non-leaf nodes to optimize (simple arrangement)")
    
    for node in non_leaf_nodes:
        if node.arrangement and node.arrangement.objs:
            logger.info(f"  Optimizing parent motif: {node.motif_type} (depth={node.depth})")
            node_start_time = time.time()
            
            # Build a combined arrangement that treats each child motif as a super-object
            # and includes parent-only objects (e.g., items not part of any child motif)
            combined_with_parent = _create_parent_combined_arrangement(node)
            if combined_with_parent and len(combined_with_parent.objs) > 1:
                optimized_combined_parent = optimize_single_arrangement(
                    combined_with_parent,
                    resolve_collisions=resolve_collisions,
                    collision_move_step=collision_move_step,
                    collision_max_iters=collision_max_iters,
                    make_tight=make_tight,
                    make_tight_iters=make_tight_iters,
                    approximate_gravity=approximate_gravity
                )
                _apply_parent_transforms(node, combined_with_parent, optimized_combined_parent)
            
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
                make_tight=make_tight,
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


def _create_parent_combined_arrangement(node):
    """Create a combined arrangement for a parent node by:
    - collapsing each child motif into a single super-object (preserving internal structure)
    - including parent-only objects (objects not contained in any child motif)
    """
    from hsm_core.scene_motif import Arrangement
    from hsm_core.scene_motif.core.obj import Obj
    from hsm_core.scene_motif.core.bounding_box import BoundingBox

    if not node or not node.arrangement:
        return None

    # 1) Build super-objects for each child motif (using their current arrangement meshes)
    child_super_objs = []
    child_labels: set[str] = set()
    for i, child in enumerate(node.children):
        if not child.arrangement or not child.arrangement.objs:
            continue
        motif_meshes = []
        for obj in child.arrangement.objs:
            # track labels to identify parent-only objects later
            if hasattr(obj, 'label'):
                child_labels.add(obj.label)
            if hasattr(obj, 'mesh') and obj.mesh is not None:
                obj_mesh = deepcopy(obj.mesh)
                if hasattr(obj, 'bounding_box') and obj.bounding_box:
                    obj_mesh.apply_transform(obj.bounding_box.no_scale_matrix)
                motif_meshes.append(obj_mesh)
        if not motif_meshes:
            continue
        combined_mesh = motif_meshes[0] if len(motif_meshes) == 1 else trimesh.util.concatenate(motif_meshes)
        bounds = combined_mesh.bounds
        centroid = combined_mesh.centroid
        half_size = (bounds[1] - bounds[0]) / 2
        super_obj = Obj(
            label=f"motif_{child.motif_type}_{i}",
            mesh=combined_mesh,
            bounding_box=BoundingBox(
                centroid=centroid,
                half_size=half_size,
                coord_axes=np.eye(3)
            )
        )
        setattr(super_obj, '_motif_node', child)
        child_super_objs.append(super_obj)

    # 2) Collect parent-only objects (those not belonging to any child motif)
    parent_extra_objs = []
    for obj in node.arrangement.objs:
        try:
            label = getattr(obj, 'label', None)
            # Heuristics: skip placeholders referencing sub-arrangements; include only real meshes
            if label and (label in child_labels or label.startswith('sub_arrangements[') or label.startswith('execute_results[')):
                continue
            if hasattr(obj, 'mesh') and obj.mesh is not None:
                # Create a shallow super-object clone using current world transform
                extra_mesh = deepcopy(obj.mesh)
                if hasattr(obj, 'bounding_box') and obj.bounding_box:
                    extra_mesh.apply_transform(obj.bounding_box.no_scale_matrix)
                from hsm_core.scene_motif.core.obj import Obj as SMObj
                from hsm_core.scene_motif.core.bounding_box import BoundingBox as SMBox
                bounds = extra_mesh.bounds
                centroid = extra_mesh.centroid
                half_size = (bounds[1] - bounds[0]) / 2
                extra_obj = SMObj(
                    label=label or "parent_obj",
                    mesh=extra_mesh,
                    bounding_box=SMBox(
                        centroid=centroid,
                        half_size=half_size,
                        coord_axes=np.eye(3)
                    )
                )
                # Keep a back-reference to the original parent object for transform distribution
                setattr(extra_obj, '_parent_ref_label', label)
                parent_extra_objs.append(extra_obj)
        except Exception:
            # Be defensive; skip any problematic object
            continue

    combined_objs = child_super_objs + parent_extra_objs
    if not combined_objs:
        return None
    return Arrangement(combined_objs, f"Parent combined for {node.motif_type}", f"combined_parent({node.motif_type})")


def _apply_parent_transforms(node, original_combined, optimized_combined) -> None:
    """Distribute transforms from an optimized combined arrangement back to:
    - child motif objects (translation applied to each child's objects)
    - parent-only objects (matched by label)
    """
    if not node or not node.arrangement:
        return
    if len(original_combined.objs) != len(optimized_combined.objs):
        logger.warning("Mismatch in object count between original and optimized parent-combined arrangements")
        # Fall back to just updating the node arrangement with no distribution
        return

    # Build a quick index for parent-only objects by label
    parent_objs_by_label = {}
    for obj in node.arrangement.objs:
        lbl = getattr(obj, 'label', None)
        if lbl:
            parent_objs_by_label.setdefault(lbl, []).append(obj)

    # Walk through paired original/optimized objects and apply translations
    for orig_obj, opt_obj in zip(original_combined.objs, optimized_combined.objs):
        # Safely fetch centroids
        orig_bb = getattr(orig_obj, 'bounding_box', None)
        opt_bb = getattr(opt_obj, 'bounding_box', None)
        orig_centroid = orig_bb.centroid if (orig_bb is not None and hasattr(orig_bb, 'centroid')) else None
        opt_centroid = opt_bb.centroid if (opt_bb is not None and hasattr(opt_bb, 'centroid')) else None
        if orig_centroid is None or opt_centroid is None:
            continue
        translation = opt_centroid - orig_centroid

        # Case A: this combined object represents a child motif
        motif_node = getattr(orig_obj, '_motif_node', None)
        if motif_node and motif_node.arrangement:
            for child_obj in motif_node.arrangement.objs:
                if hasattr(child_obj, 'bounding_box') and child_obj.bounding_box is not None:
                    child_obj.bounding_box.centroid += translation
            # Persist back the updated child arrangement into the hierarchy
            # (node already references motif_node in the hierarchy tree)
            continue

        # Case B: this combined object represents an extra parent-level object
        parent_ref_label = getattr(orig_obj, '_parent_ref_label', None)
        if parent_ref_label and parent_ref_label in parent_objs_by_label:
            for target in parent_objs_by_label[parent_ref_label]:
                if hasattr(target, 'bounding_box') and target.bounding_box is not None:
                    target.bounding_box.centroid += translation

    # No need to replace node.arrangement; we updated objects in place
    # Ensure hierarchy remains consistent
    return


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