"""
Result Handlers Module

This module handles the application and logging of retrieval results.
"""

import logging
from typing import List, Dict
from pathlib import Path
from copy import deepcopy

from hsm_core.scene_motif.core.obj import Obj
from hsm_core.retrieval.utils.retriever_helpers import TransformTracker

logger = logging.getLogger(__name__)


def apply_and_log_results(
    all_objs: List[Obj],
    processed_objs: List[Obj],
    mesh_dict: Dict[str, Dict],
    same_per_label: bool = True
) -> None:
    """
    Applies cached meshes to duplicates and logs the final results.

    Args:
        all_objs: Complete list of all objects
        processed_objs: List of objects that were processed
        mesh_dict: Dictionary of assigned meshes
        same_per_label: Whether to use same mesh per label
    """
    # Apply stored meshes to remaining objects if using same_per_label
    if same_per_label:
        logger.info("Applying meshes for same_per_label objects")

        for obj_to_update_label in all_objs:
            # Skip if this object was one of the unique ones already processed
            is_in_objs_to_process = False
            for processed_unique_obj in processed_objs:
                if obj_to_update_label is processed_unique_obj:
                    is_in_objs_to_process = True
                    break

            if is_in_objs_to_process:
                continue

            if obj_to_update_label.label in mesh_dict:
                cached_info = mesh_dict[obj_to_update_label.label]
                obj_to_update_label.mesh = deepcopy(cached_info["mesh"])
                obj_to_update_label.mesh_path = cached_info["path"]

                # Initialize transform tracker and copy transforms from the original object
                if obj_to_update_label.transform_tracker is None:
                    obj_to_update_label.transform_tracker = TransformTracker()

                # Find the original object that was processed to copy its transforms
                for processed_obj in processed_objs:
                    if processed_obj.label == obj_to_update_label.label and processed_obj.transform_tracker:
                        # Copy transforms from the processed object
                        mesh_id = str(Path(obj_to_update_label.mesh_path).stem)
                        original_transforms = processed_obj.transform_tracker.get_transforms(mesh_id)
                        for transform in original_transforms:
                            obj_to_update_label.transform_tracker.add_transform(
                                mesh_id,
                                transform.transform_type,
                                transform.transform_matrix,
                                **transform.metadata
                            )
                        break

                obj_to_update_label.bounding_box.half_size = obj_to_update_label.mesh.bounding_box_oriented.extents / 2
                logger.debug(f"Applied cached mesh {Path(obj_to_update_label.mesh_path).name} to duplicate object {obj_to_update_label.label}. BBox: {obj_to_update_label.bounding_box.half_size}")
            else:
                logger.warning(f"No cached mesh found for label '{obj_to_update_label.label}'. Duplicate object {obj_to_update_label.label} (index {all_objs.index(obj_to_update_label)}) will have no mesh.")
                obj_to_update_label.mesh = None
                obj_to_update_label.mesh_path = None

    # Log results
    log_entries = []
    for obj_log_iter in all_objs:
        path_str = Path(obj_log_iter.mesh_path).name if obj_log_iter.mesh_path else "No mesh found"
        bbox_str = str(obj_log_iter.bounding_box.half_size * 2) if obj_log_iter.mesh else "N/A"

        rotation_str = "N/A"
        if obj_log_iter.label in mesh_dict and mesh_dict[obj_log_iter.label].get("mesh") is not None:
            rotation_str = str(mesh_dict[obj_log_iter.label].get("rotation_info", "N/A"))

        log_entries.append(f"- {obj_log_iter.label}: {path_str} (BBox: {bbox_str}, Rotation: {rotation_str})")

    logger.info("Retrieved Meshes Summary:")
    for entry in log_entries:
        logger.info(entry)
