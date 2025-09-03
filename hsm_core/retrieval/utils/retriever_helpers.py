#!/usr/bin/env python3
"""
Retrieval Helper Functions
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import trimesh
import logging
from pathlib import Path
from copy import deepcopy

from hsm_core.scene_motif.core.obj import Obj
from hsm_core.retrieval.utils.transform_tracker import TransformTracker, TransformInfo

logger = logging.getLogger(__name__)


def _ensure_transform_tracker(obj: Obj) -> TransformTracker:
    """Ensure the object has a transform tracker initialized."""
    if obj.transform_tracker is None:
        obj.transform_tracker = TransformTracker()
    return obj.transform_tracker


def _calculate_bbox_score(target_dimensions: np.ndarray, mesh_extents: np.ndarray) -> float:
    """Calculate bounding box score based on dimension differences."""
    return np.sum(np.abs(target_dimensions - mesh_extents))


def _apply_height_penalty(score: float, mesh_extents: np.ndarray, max_height: float, penalty_threshold: float = 0.95) -> tuple[float, bool]:
    """Apply height penalty to score if mesh exceeds height limit."""
    if max_height != -1.0 and mesh_extents[1] > max_height * penalty_threshold:
        return score + 1000.0, True
    return score, False


def apply_hssd_alignment_transform(
    mesh: trimesh.Trimesh,
    mesh_id: str,
    hssd_alignment_data: Dict[str, Any] | List[Any],
    obj: Obj
) -> Tuple[trimesh.Trimesh, bool]:
    """Apply HSSD alignment transformation to a mesh if available.
    
    Args:
        mesh: The `trimesh.Trimesh` to transform.
        mesh_id: The unique mesh identifier (usually the GLB filename stem).
        hssd_alignment_data: Parsed JSON data from *hssd_wnsynsetkey_index.json*.
        obj: The parent :class:`hsm_core.objects.obj.Obj` – required so the
            generated :class:`TransformInfo` can be stored in its
            :class:`TransformTracker`.

    Returns:
        Tuple of (transformed_mesh, transform_applied)
    """

    if not hssd_alignment_data:
        return mesh, False

    def _iter_entries(data: Any):  # type: ignore[override]
        """Yield dict entries containing at least *id*, *up*, *front*."""
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    yield entry
        elif isinstance(data, dict):
            for value in data.values():
                # Values can be lists or nested dicts – recurse once.
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict):
                            yield entry
                elif isinstance(value, dict):
                    yield value

    # Normalise *mesh_id* to string for comparisons.
    target_mesh_id = str(mesh_id)
    transform_tracker = _ensure_transform_tracker(obj)

    for entry in _iter_entries(hssd_alignment_data):
        if entry.get("id") != target_mesh_id:
            continue

        up_vec: str | None = entry.get("up")
        front_vec: str | None = entry.get("front")

        if not up_vec or not front_vec:
            return mesh, False

        transform_matrix = create_rotation_matrix(up_vec, front_vec)
        if transform_matrix is None:
            logger.debug(f"[HSSD] Orientation of mesh {target_mesh_id} already canonical or invalid – no transform applied")
            return mesh, False

        # Apply and record the transform.
        mesh.apply_transform(transform_matrix)

        transform_tracker.add_transform(
            target_mesh_id,
            "hssd_alignment",
            transform_matrix,
            up=up_vec,
            front=front_vec,
        )

        logger.debug(f"[HSSD] Applied alignment transform to mesh {target_mesh_id} (up={up_vec}, front={front_vec})")

        return mesh, True

    # No entry found for mesh_id → no transform.
    return mesh, False


def create_rotation_matrix(up: str, front: str) -> np.ndarray | None:
    """Create a 4×4 rotation matrix from up and front vectors.

    If the given orientation already matches the canonical target (front
    = +Z, up = +Y) the function returns None so callers can skip the
    transform.
    """

    import re

    def _parse_vec(vec_str: str) -> np.ndarray | None:
        """Return a 3-element numpy array or *None* on failure."""
        cleaned = vec_str.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        # Split on comma or any whitespace
        parts = re.split(r"[\s,]+", cleaned)
        parts = [p for p in parts if p]
        if len(parts) != 3:
            return None
        try:
            return np.array([float(p) for p in parts], dtype=float)
        except ValueError:
            return None

    up_array = _parse_vec(up)
    front_array = _parse_vec(front)

    if up_array is None or front_array is None:
        return None

    target_front = np.array([0, 0, 1])
    target_up = np.array([0, 1, 0])

    if np.allclose(up_array, target_up) and np.allclose(front_array, target_front):
        return None

    # Normalize input vectors
    up_norm = np.linalg.norm(up_array)
    front_norm = np.linalg.norm(front_array)

    if up_norm < 1e-6 or front_norm < 1e-6:
        return None

    norm_up: np.ndarray = up_array / up_norm
    norm_front: np.ndarray = front_array / front_norm

    if np.allclose(norm_front, norm_up) or np.allclose(norm_front, -norm_up):
        return None

    right: np.ndarray = np.cross(norm_front, norm_up)

    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        return None

    norm_right: np.ndarray = right / right_norm
    target_right = np.cross(target_front, target_up)

    # Create source and target matrices for initial alignment
    source_matrix = np.column_stack((norm_right, norm_front, norm_up))
    target_matrix = np.column_stack((target_right, target_front, target_up))

    # Calculate initial rotation matrix that transforms from source to target
    initial_rotation_matrix = target_matrix @ source_matrix.T

    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = initial_rotation_matrix

    return transform


def optimize_mesh_rotation(
    obj: Obj,
    mesh: trimesh.Trimesh,
    mesh_id: str,
    try_side_rotations: bool = False,
    max_height: float = -1.0,
    max_height_penalty_percentage: float = 0.95
) -> Tuple[trimesh.Trimesh, float, Dict[str, Any], bool]:
    """Optimize the rotation of the mesh for the object based on bounding box fit."""
    # Initialize transform tracker for this object if not present
    transform_tracker = _ensure_transform_tracker(obj)
    
    RADIAN_90 = np.radians(90)
    
    side_rotations = [(0, [1, 0, 0])]
    
    target_dimensions = obj.bounding_box.full_size
    
    best_score = np.inf
    best_mesh = None
    best_rotation_info: Dict[str, Any] = {}
    best_is_penalized = False
    
    for side_angle, side_axis in side_rotations:
        rotated_mesh_side = deepcopy(mesh)
        if side_angle != 0:
            side_transform = trimesh.transformations.rotation_matrix(side_angle, side_axis)
            rotated_mesh_side.apply_transform(side_transform)
        
        y_rotation_steps = [0, -1]
        
        for y_rotation_step in y_rotation_steps:
            current_mesh = deepcopy(rotated_mesh_side)
            if y_rotation_step != 0:
                y_rotation_matrix = trimesh.transformations.rotation_matrix(
                    y_rotation_step * RADIAN_90, [0, 1, 0]
                )
                current_mesh.apply_transform(y_rotation_matrix)
            
            mesh_extents = current_mesh.extents
            score = _calculate_bbox_score(target_dimensions, mesh_extents)
            score, current_is_penalized = _apply_height_penalty(score, mesh_extents, max_height, max_height_penalty_percentage)
            
            if score < best_score:
                best_score = score
                best_mesh = current_mesh
                best_rotation_info = {
                    "side_rotation": f"{np.degrees(side_angle):.0f}° around {side_axis}",
                    "y_rotation": f"{y_rotation_step * 90}°",
                }
                best_is_penalized = current_is_penalized
    
    if best_mesh is None:
        best_mesh = mesh
        mesh_extents_original = mesh.extents
        best_score = _calculate_bbox_score(target_dimensions, mesh_extents_original)
        best_rotation_info = {
            "side_rotation": "0° around [1 0 0]",
            "y_rotation": "0°",
            "optimization_skipped_no_better_found": True
        }
        
        best_score, best_is_penalized = _apply_height_penalty(best_score, mesh_extents_original, max_height, max_height_penalty_percentage)
    
    # Track the rotation optimization in the object's tracker
    if best_rotation_info and not best_rotation_info.get("optimization_skipped_no_better_found", False):
        transform_tracker.add_transform(
            mesh_id,
            "rotation_optimization",
            metadata=best_rotation_info
        )
    
    return best_mesh, best_score, best_rotation_info, best_is_penalized


def load_and_normalize_mesh(mesh_path: Path, obj: Obj) -> trimesh.Trimesh:
    """Load mesh and normalize its position (center XZ, bottom at Y=0)."""
    # Initialize transform tracker for this object if not present
    transform_tracker = _ensure_transform_tracker(obj)
    
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    
    # Get the bounding box bounds
    bbox = mesh.bounds
    
    # Calculate translation to center XZ and set bottom to Y=0
    translation = [
        -(bbox[0][0] + bbox[1][0]) / 2,  # Center X
        -bbox[0][1],                    # Move bottom to Y=0
        -(bbox[0][2] + bbox[1][2]) / 2   # Center Z
    ]
    
    # Translate mesh
    mesh.apply_translation(translation)
    
    # Track the normalization in the object's tracker
    mesh_id = str(mesh_path.stem)
    transform_tracker.add_transform(
        mesh_id,
        "normalization",
        metadata={"translation": translation}
    )
    
    return mesh


def validate_support_surface_constraints(
    obj: Obj,
    mesh: trimesh.Trimesh,
    support_surface_data: Optional[Dict] = None,
    max_surface_occupancy: float = 0.8
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that a mesh fits within the available support surface constraints."""
    constraint_info = {
        # "fits_surface": True,
        # "surface_occupancy": 0.0,
        "height_clearance": True,
        # "overhang_ratio": 0.0
    }
    
    if not support_surface_data:
        return True, constraint_info
    
    mesh_extents = mesh.extents
    # obj_footprint_area = mesh_extents[0] * mesh_extents[2]
    
    # # -------------------------------------------------------------
    # # 1. Surface area / occupancy checks
    # # -------------------------------------------------------------
    # # Support both legacy key names ("available_area") and the newer
    # # ones created in `Scene._create_support_surface_constraints`
    # # ("max_area", or explicit width/depth bounds).
    # available_surface_area: float = support_surface_data.get("available_area", float("inf"))
    # if available_surface_area == float("inf"):
    #     # Try to derive it from max_area or max_width*max_depth
    #     max_area = support_surface_data.get("max_area")
    #     if isinstance(max_area, (int, float)) and max_area > 0:
    #         available_surface_area = max_area
    #     else:
    #         # Build from explicit width / depth if given
    #         max_w = support_surface_data.get("max_width")
    #         max_d = support_surface_data.get("max_depth")
    #         if isinstance(max_w, (int, float)) and isinstance(max_d, (int, float)):
    #             available_surface_area = max_w * max_d

    # if available_surface_area != float("inf"):
    #     surface_occupancy = obj_footprint_area / max(available_surface_area, 1e-6)
    #     constraint_info["surface_occupancy"] = surface_occupancy
    #     if surface_occupancy > max_surface_occupancy:
    #         constraint_info["fits_surface"] = False

    # -------------------------------------------------------------
    # 2. Height clearance
    # -------------------------------------------------------------
    max_height_allowed = support_surface_data.get("max_height", support_surface_data.get("height_limit", float("inf")))
    if max_height_allowed != float("inf") and mesh_extents[1] > max_height_allowed:
        constraint_info["height_clearance"] = False

    # # -------------------------------------------------------------
    # # 3. Overhang constraints
    # # -------------------------------------------------------------
    # surface_bounds = support_surface_data.get("bounds") or support_surface_data.get("surface_bounds")
    # if surface_bounds:
    #     if "width" in surface_bounds and "depth" in surface_bounds:
    #         surface_width = surface_bounds.get("width", float("inf"))
    #         surface_depth = surface_bounds.get("depth", float("inf"))
    #     else:
    #         # Bounds given as min/max coordinates; derive extents
    #         min_pt = surface_bounds.get("min")
    #         max_pt = surface_bounds.get("max")
    #         if min_pt and max_pt and len(min_pt) == 2 and len(max_pt) == 2:
    #             surface_width = max_pt[0] - min_pt[0]
    #             surface_depth = max_pt[1] - min_pt[1]
    #         else:
    #             surface_width = surface_depth = float("inf")

    #     width_overhang = max(0, mesh_extents[0] - surface_width) / mesh_extents[0]
    #     depth_overhang = max(0, mesh_extents[2] - surface_depth) / mesh_extents[2]
    #     constraint_info["overhang_ratio"] = max(width_overhang, depth_overhang)

    #     if constraint_info["overhang_ratio"] > 0.1:
    #         constraint_info["fits_surface"] = False
    
    # is_valid = (constraint_info["fits_surface"] and 
    #             constraint_info["height_clearance"] and 
    #             constraint_info["overhang_ratio"] <= 0.3)
    
    return True, constraint_info

def process_mesh_candidate(
    obj: Obj,
    mesh_path: Path,
    mesh_id: str,
    object_type,
    hssd_alignment_data: Dict[str, Any],
    max_height: float = -1.0,
    support_surface_constraints: Optional[Dict] = None
) -> Optional[Tuple[float, trimesh.Trimesh, Path, str, Dict[str, Any], bool, Dict[str, Any]]]:
    """Process a single mesh candidate and return evaluation results."""
    try:
        # Load and normalize mesh
        loaded_mesh = load_and_normalize_mesh(mesh_path, obj)
        
        # Apply HSSD alignment if applicable
        hssd_transform_applied = False
        loaded_mesh, hssd_transform_applied = apply_hssd_alignment_transform(
            loaded_mesh, mesh_id, hssd_alignment_data, obj
        )

        # Optimize rotation or use HSSD-transformed mesh
        if hssd_transform_applied:
            optimized_mesh = loaded_mesh
            target_dimensions = obj.bounding_box.full_size
            mesh_extents = optimized_mesh.bounding_box_oriented.extents
            bb_score = _calculate_bbox_score(target_dimensions, mesh_extents)
            bb_score, is_penalized = _apply_height_penalty(bb_score, mesh_extents, max_height, 0.95)

            rotation_info = {"hssd_applied": True, "optimization_skipped": True}

            logger.debug(f"Skipping rotation optimization for {mesh_id}, using HSSD alignment. Score: {bb_score:.4f}, Penalized: {is_penalized}")
        else:
            optimized_mesh, bb_score, rotation_info, is_penalized = optimize_mesh_rotation(
                obj, loaded_mesh, mesh_id, False, max_height
            )
            rotation_info["hssd_applied"] = False
        
        # Validate support surface constraints
        constraint_info: Dict[str, Any] = {}
        if support_surface_constraints and obj.label in support_surface_constraints:
            support_constraint_valid, constraint_info = validate_support_surface_constraints(
                obj, optimized_mesh, support_surface_constraints[obj.label]
            )
            if not support_constraint_valid:
                bb_score += 500.0
                is_penalized = True
                logger.debug(f"Support constraint violation for {obj.label}: {constraint_info}")

        return (bb_score, optimized_mesh, mesh_path, mesh_id, rotation_info, is_penalized, constraint_info)

    except Exception as e:
        logger.warning(f"Failed to process mesh {mesh_path} for {obj.label}: {e}")
        return None


def sort_candidates_by_quality(candidates: List[Tuple]) -> List[Tuple]:
    """Sort candidates by penalized status (False first), then by bounding box score (ascending)."""
    return sorted(candidates, key=lambda x: (x[5], x[0]))  # x[5] is is_penalized, x[0] is bb_score


def apply_mesh_to_object(
    obj: Obj,
    selected_mesh: trimesh.Trimesh,
    selected_path: str,
    selected_mesh_id: str
) -> None:
    """Apply the selected mesh to an object and update its properties."""
    obj.mesh = selected_mesh
    obj.mesh_path = str(selected_path)
    obj.bounding_box.half_size = obj.mesh.bounding_box_oriented.extents / 2
    logger.debug(f"Applied mesh to {obj.label}: BBox Half Size: {obj.bounding_box.half_size}") 