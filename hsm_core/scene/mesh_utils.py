"""Mesh loading and preprocessing utilities for 3D scene objects."""

import logging
import trimesh
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import os

from hsm_core.scene.objects import SceneObject

logger = logging.getLogger(__name__)


def _load_mesh_standard(obj: SceneObject, verbose: bool) -> Tuple[Optional[trimesh.Trimesh], str]:
    """Load mesh using standard trimesh loading strategy."""
    try:
        loaded_obj = trimesh.load(obj.mesh_path, force="mesh", process=False)
        if verbose:
            logger.debug(f"Loaded {obj.name} using standard force='mesh' strategy")
        return loaded_obj, "standard_force_mesh"
    except ValueError as e:
        if "multiple scenes" in str(e).lower():
            return _load_mesh_multiscene(obj, verbose)
        if verbose:
            logger.debug(f"Standard loading failed for {obj.name}: {e}")
        return None, ""
    except Exception as e:
        if verbose:
            logger.debug(f"Standard loading failed for {obj.name}: {e}")
        return None, ""


def _load_mesh_multiscene(obj: SceneObject, verbose: bool) -> Tuple[Optional[trimesh.Trimesh], str]:
    """Handle multi-scene GLBs by loading without force parameter."""
    try:
        scene_or_mesh = trimesh.load(obj.mesh_path, process=False)
        if hasattr(scene_or_mesh, 'geometry') and scene_or_mesh.geometry:
            # Extract first mesh from scene
            first_geom = next(iter(scene_or_mesh.geometry.values()))
            if hasattr(first_geom, 'vertices'):
                if verbose:
                    logger.debug(f"Loaded {obj.name} using multi-scene extraction strategy")
                return first_geom, "multi_scene_extraction"
        elif hasattr(scene_or_mesh, 'vertices'):
            # Direct mesh object
            if verbose:
                logger.debug(f"Loaded {obj.name} using direct mesh strategy")
            return scene_or_mesh, "direct_mesh"
        raise ValueError("Loaded object is neither scene nor mesh")
    except Exception as e:
        if verbose:
            logger.debug(f"Multi-scene loading failed for {obj.name}: {e}")
        return None, ""


def _parse_rotation_degrees(value: str) -> float:
    """Parse rotation value from string format (e.g., '45°', '0°')."""
    try:
        return float(value.split('°')[0].strip())
    except Exception:
        return 0.0


def _parse_rotation_axis(side_raw: str) -> List[float]:
    """Parse rotation axis from string format."""
    if 'around' not in side_raw:
        return [1.0, 0.0, 0.0]

    axis_txt = side_raw.split('around', 1)[1].strip().replace('[', '').replace(']', '')
    vals = [v for v in axis_txt.split() if v]
    if len(vals) == 3:
        try:
            return [float(v) for v in vals]
        except ValueError:
            pass
    return [1.0, 0.0, 0.0]


def _apply_rotation_optimization(obj: SceneObject, loaded_obj: trimesh.Trimesh, verbose: bool) -> None:
    """Apply rotation optimization transform to the loaded mesh."""

    try:
        rot_opt_info = obj.get_transform('rotation_optimization')  # type: ignore[attr-defined]
    except Exception:
        return

    if rot_opt_info is None or not hasattr(rot_opt_info, 'metadata'):
        return

    try:
        side_raw = str(rot_opt_info.metadata.get('side_rotation', '0°'))
        y_raw = str(rot_opt_info.metadata.get('y_rotation', '0°'))

        side_deg = _parse_rotation_degrees(side_raw)
        y_deg = _parse_rotation_degrees(y_raw)
        side_axis = _parse_rotation_axis(side_raw)

        # Apply side rotation if significant
        if abs(side_deg) > 1e-3:
            side_mat = trimesh.transformations.rotation_matrix(
                angle=np.radians(side_deg),
                direction=side_axis,
                point=[0, 0, 0]
            )
            loaded_obj.apply_transform(side_mat)
        else:
            side_mat = np.eye(4)

        # Apply Y rotation if significant
        if abs(y_deg) > 1e-3:
            y_mat = trimesh.transformations.rotation_matrix(
                angle=np.radians(y_deg),
                direction=[0, 1, 0],
                point=[0, 0, 0]
            )
            loaded_obj.apply_transform(y_mat)
        else:
            y_mat = np.eye(4)

        # Store transformation data
        rot_comb = y_mat @ side_mat
        obj._preprocessing_data = getattr(obj, '_preprocessing_data', {})
        obj._preprocessing_data.update({
            'rotation_optimization_matrix': rot_comb.tolist(),
            'rotation_optimization': rot_opt_info.metadata
        })

        if verbose:
            logger.debug(f"Applied rotation optimisation to {obj.name}: side {side_deg:.1f}°, Y {y_deg:.1f}°")

    except Exception as rot_err:
        logger.warning(f"Rotation optimisation application failed for {obj.name}: {rot_err}")
        if verbose:
            logger.warning(f"Rotation optimisation application failed for {obj.name}: {rot_err}")


def _load_hssd_cache() -> Dict[str, Dict]:
    """Load and cache HSSD index data."""
    cache_key = '_HSSD_INDEX_CACHE'
    if cache_key not in globals():
        globals()[cache_key] = None

    hssd_cache = globals()[cache_key]
    if hssd_cache is not None:
        return hssd_cache

    index_path = Path(__file__).parents[2] / 'data' / 'preprocessed' / 'hssd_wnsynsetkey_index.json'
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    import json as _json
    with open(index_path, 'r') as f:
        raw_data = _json.load(f)

    def _iter_entries(data):
        if isinstance(data, list):
            for e in data:
                if isinstance(e, dict):
                    yield e
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    for e in v:
                        if isinstance(e, dict):
                            yield e
                elif isinstance(v, dict):
                    yield v

    hssd_cache = {e.get('id'): e for e in _iter_entries(raw_data)}
    globals()[cache_key] = hssd_cache
    return hssd_cache


def _apply_primary_hssd_alignment(obj: SceneObject, loaded_obj: trimesh.Trimesh,
                                  loading_strategy: str, verbose: bool) -> bool:
    """Apply primary HSSD alignment from transform tracker."""
    if not obj.has_hssd_alignment():
        return False

    try:
        hssd_transform_info = obj.get_hssd_alignment_transform()
        if hssd_transform_info and hssd_transform_info.transform_matrix is not None:
            if verbose:
                logger.debug(f"Applying HSSD alignment transform to {obj.name}")

            loaded_obj.apply_transform(hssd_transform_info.transform_matrix)

            # Store the transform matrix in preprocessing data for later use
            obj._preprocessing_data = getattr(obj, '_preprocessing_data', {})
            obj._preprocessing_data.update({
                'transform_matrix': hssd_transform_info.transform_matrix,
                'loading_strategy': loading_strategy
            })

            if verbose:
                logger.debug(f"HSSD transform applied successfully to {obj.name}")
            return True
        elif verbose:
            logger.warning(f"{obj.name} has HSSD alignment but no transform matrix found")
    except Exception as e:
        logger.warning(f"HSSD transform application failed for {obj.name}: {e}")
        if verbose:
            logger.warning(f"HSSD transform application failed for {obj.name}: {e}")

    return False


def _apply_fallback_hssd_alignment(obj: SceneObject, loaded_obj: trimesh.Trimesh, verbose: bool) -> None:
    """Apply fallback HSSD alignment using cached index data."""
    try:
        from hsm_core.retrieval.utils.retriever_helpers import create_rotation_matrix
        from hsm_core.retrieval.utils.transform_tracker import TransformInfo

        hssd_cache = _load_hssd_cache()
        mesh_id = Path(obj.mesh_path).stem if obj.mesh_path else ''
        entry = hssd_cache.get(mesh_id)

        if entry:
            up_vec = entry.get('up')
            front_vec = entry.get('front')
            if up_vec and front_vec:
                rot_mat = create_rotation_matrix(up_vec, front_vec)
                if rot_mat is not None:
                    loaded_obj.apply_transform(rot_mat)

                    # Record transform
                    transform_info = TransformInfo(
                        transform_type='hssd_alignment',
                        transform_matrix=rot_mat,
                        metadata={'up': up_vec, 'front': front_vec},
                        applied_order=0
                    )
                    obj.add_transform('hssd_alignment', transform_info)

                    obj._preprocessing_data = getattr(obj, '_preprocessing_data', {})
                    obj._preprocessing_data['transform_matrix'] = rot_mat

                    if verbose:
                        logger.debug(f"Applied fallback HSSD alignment to {obj.name}")

    except Exception as idx_err:
        logger = logging.getLogger(__name__)
        logger.warning(f"Fallback HSSD alignment failed for {obj.name}: {idx_err}")
        if verbose:
            logger.warning(f"Fallback HSSD alignment failed for {obj.name}: {idx_err}")


def _apply_hssd_alignment(obj: SceneObject, loaded_obj: trimesh.Trimesh,
                          loading_strategy: str, verbose: bool) -> None:
    """Apply HSSD alignment transforms to the loaded mesh."""
    # Try primary HSSD alignment first
    if not _apply_primary_hssd_alignment(obj, loaded_obj, loading_strategy, verbose):
        # If primary fails or doesn't exist, try fallback
        if verbose and not obj.has_hssd_alignment():
            logger.debug(f"No HSSD alignment transform found for {obj.name}")
        _apply_fallback_hssd_alignment(obj, loaded_obj, verbose)


def _apply_mesh_mirroring(obj: SceneObject, loaded_obj: trimesh.Trimesh, verbose: bool) -> None:
    """Apply X-axis mirroring to the mesh."""
    logger = logging.getLogger(__name__)

    try:
        centroid = loaded_obj.centroid
        mirror_mat = trimesh.transformations.reflection_matrix(
            point=centroid,
            normal=[1, 0, 0]
        )
        loaded_obj.apply_transform(mirror_mat)

        # Fix normals if possible
        try:
            loaded_obj.fix_normals()
        except Exception:
            pass  # Ignore if normals cannot be fixed; Trimesh will auto-handle most cases

        # Store transformation data
        obj._preprocessing_data = getattr(obj, '_preprocessing_data', {})
        obj._preprocessing_data['mirror_transform'] = mirror_mat.tolist()

        if verbose:
            logger.debug(f"Applied X-axis mirroring to {obj.name}")

    except Exception as mirror_err:
        logger.warning(f"Mirroring failed for {obj.name}: {mirror_err}")
        if verbose:
            logger.warning(f"Mirroring failed for {obj.name}: {mirror_err}")


def preprocess_object_mesh(obj: SceneObject, verbose: bool = False) -> Optional[trimesh.Trimesh]:
    """
    Standalone function to preprocess object mesh by loading, applying HSSD transforms, and normalizing.

    This function implements robust mesh loading with multiple fallback strategies:
    1. Standard trimesh.load with force="mesh"
    2. Alternative loading without force parameter for multi-scene GLBs
    3. Fallback to bounding box extraction for corrupted/unsupported files

    Args:
        obj: SceneObject to preprocess
        verbose: Whether to print debug information

    Returns:
        Preprocessed trimesh object or None if all loading strategies fail
    """
    if verbose:
        logger.info(f"Preprocessing mesh for {obj.name} ({obj.obj_type.name})")

    # Validate mesh path exists
    if not obj.mesh_path or not os.path.exists(obj.mesh_path):
        error_msg = f"Mesh file not found: {obj.mesh_path}"
        logger.error(error_msg)
        if verbose:
            print(f"ERROR: {error_msg}")
        return None

    # Try loading strategies in order of preference
    loading_strategies = [_load_mesh_standard]

    loaded_obj = None
    loading_strategy = "unknown"

    for strategy_func in loading_strategies:
        loaded_obj, strategy_name = strategy_func(obj, verbose)
        if loaded_obj is not None:
            loading_strategy = strategy_name
            break

    # Validate loaded mesh
    if loaded_obj is None or not hasattr(loaded_obj, 'vertices') or len(loaded_obj.vertices) == 0:
        error_msg = f"Loaded mesh for {obj.name} is invalid or empty"
        logger.error(error_msg)
        if verbose:
            print(f"ERROR: {error_msg}")
        return None

    _apply_rotation_optimization(obj, loaded_obj, verbose)
    _apply_hssd_alignment(obj, loaded_obj, loading_strategy, verbose)

    # Store loading strategy for debugging
    obj._preprocessing_data = getattr(obj, '_preprocessing_data', {})
    obj._preprocessing_data['loading_strategy'] = loading_strategy

    _apply_mesh_mirroring(obj, loaded_obj, verbose)

    if verbose:
        logger.info(f"Successfully preprocessed {obj.name} using {loading_strategy} strategy")

    return loaded_obj
