import numpy as np
import trimesh
from matplotlib import pyplot as plt
import matplotlib.collections
from typing import List, Tuple, Optional, Dict, Any
from hsm_core.utils import get_logger

from hsm_core.support_region.constants import MIN_AREA, MIN_DIMENSION, MAX_FACES_PER_BATCH
from hsm_core.support_region.utils import shrink_bounds
from hsm_core.support_region.geometry import segment_geometry

logger = get_logger('visualization.geometry_2d_visualizer')

def visualize_object_geometry(
    geometries: List[trimesh.Trimesh],
    ax: Optional[plt.Axes],
    color: Optional[str],
    label: Optional[str] = None,
    verbose: bool = True,
    alpha: float = 1.0,
    fill: bool = True,
    x_bounds: Optional[Tuple[float, float]] = None,
    transform_matrix: np.ndarray = np.array([[1, 0], [0, 1]]),
    shrink_factor: float = 0.0,
    view_axes_indices: List[int] = [0, 2],
    show_border: bool = True,
    min_dimension_threshold: Optional[float] = None
) -> Tuple[Optional[List[Dict[str, Any]]], List[trimesh.Trimesh]]:
    """Visualize all geometries in a given scene in 2D."""
    dimension_threshold = min_dimension_threshold if min_dimension_threshold is not None else MIN_DIMENSION

    if verbose:
        logger.debug("Starting visualize_object_geometry")
        logger.debug(f"Input parameters: len={len(geometries)} color={color}, label={label}, x_bounds={x_bounds}, shrink_factor={shrink_factor}, view_axes={view_axes_indices}")

    transformed_geometries = []
    seen_bounds = set()

    for geometry in geometries:
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'area'):
             if verbose: logger.debug(f"Skipping invalid geometry object: {type(geometry)}")
             continue
        new_geom = geometry.copy()

        vertices_2d = new_geom.vertices[:, view_axes_indices]
        vertices_2d = np.dot(vertices_2d, transform_matrix)

        total_area = new_geom.area
        if total_area < MIN_AREA:
            if verbose:
                logger.debug(f"Skipping geometry with small 3D area: {total_area:.3f}m²")
            continue

        bounds = tuple(np.concatenate([
            np.min(vertices_2d, axis=0),
            np.max(vertices_2d, axis=0)
        ]).round(4))

        if bounds not in seen_bounds:
            seen_bounds.add(bounds)
            transformed_geometries.append(new_geom)

    if verbose and len(transformed_geometries) < len(geometries):
        logger.debug(f"Removed {len(geometries) - len(transformed_geometries)} duplicate/small geometries")

    if x_bounds is not None and view_axes_indices[0] == 0:
        logger.debug("x_bounds provided, segmenting geometries based on X-axis")
        x_min, x_max = x_bounds
        if shrink_factor > 0:
            x_min_arr, x_max_arr = shrink_bounds(np.array(x_min), np.array(x_max), shrink_factor)
            x_min, x_max = x_min_arr.item(), x_max_arr.item()

        filtered_geometries = []
        for geom in transformed_geometries:
            segmented_geom = segment_geometry(geom, x_min, x_max, verbose=verbose)
            if segmented_geom is not None:
                vertices_2d = segmented_geom.vertices[:, view_axes_indices]
                vertices_2d = np.dot(vertices_2d, transform_matrix)
                min_b = np.min(vertices_2d, axis=0)
                max_b = np.max(vertices_2d, axis=0)
                extents = max_b - min_b

                if extents[0] < dimension_threshold or extents[1] < dimension_threshold:
                    if verbose:
                        logger.debug(f"Filtered out segmented geometry: dimensions too small -> extents: {extents[0]:.3f}m, {extents[1]:.3f}m")
                    continue

                filtered_geometries.append(segmented_geom)

        transformed_geometries = filtered_geometries
        if verbose:
            logger.debug(f"Segmented geometries between x=[{x_min:.3f}, {x_max:.3f}], got {len(transformed_geometries)} pieces")
    elif x_bounds is not None:
         logger.warning("x_bounds provided but X-axis (index 0) is not the first view axis. Skipping segmentation.")

    if not transformed_geometries:
        if verbose:
            logger.debug("No geometries after filtering")
        return None, []

    if ax:
        patch_collection = []
        total_faces = 0

        for geometry in transformed_geometries:
            if not hasattr(geometry, 'faces'):
                if verbose: logger.debug(f"Skipping geometry without faces: {type(geometry)}")
                continue

            vertices_2d = geometry.vertices[:, view_axes_indices]
            vertices_2d = np.dot(vertices_2d, transform_matrix)

            if shrink_factor > 0:
                min_bounds = np.min(vertices_2d, axis=0)
                max_bounds = np.max(vertices_2d, axis=0)
                extent = max_bounds - min_bounds
                if extent[0] > 0 and extent[1] > 0:
                    new_min_1, new_max_1 = shrink_bounds(min_bounds[0], max_bounds[0], shrink_factor)
                    new_min_2, new_max_2 = shrink_bounds(min_bounds[1], max_bounds[1], shrink_factor)
                    center = (min_bounds + max_bounds) / 2
                    scale_1 = (new_max_1 - new_min_1) / extent[0]
                    scale_2 = (new_max_2 - new_min_2) / extent[1]

                    vertices_2d = vertices_2d.copy()
                    vertices_2d[:, 0] = (vertices_2d[:, 0] - center[0]) * scale_1 + center[0]
                    vertices_2d[:, 1] = (vertices_2d[:, 1] - center[1]) * scale_2 + center[1]

            for face in geometry.faces:
                triangle = vertices_2d[face]
                v1, v2, v3 = triangle
                area = 0.5 * abs(v1[0]*(v2[1] - v3[1]) + v2[0]*(v3[1] - v1[1]) + v3[0]*(v1[1] - v2[1]))
                if area < 1e-6:
                     if verbose: logger.debug("Skipping degenerate triangle")
                     continue

                patch_collection.append(plt.Polygon(triangle, alpha=alpha, facecolor=color if fill else 'none',
                                                    edgecolor='black' if show_border else 'none',
                                                    linewidth=1.0 if show_border else 0))
                total_faces += 1

                if len(patch_collection) >= MAX_FACES_PER_BATCH:
                    if verbose:
                        logger.debug(f"Adding batch of {len(patch_collection)} polygons")
                    try:
                        collection = matplotlib.collections.PatchCollection(patch_collection, match_original=False)
                        collection.set_facecolor(color if fill else 'none')
                        collection.set_alpha(alpha)
                        if show_border:
                            collection.set_edgecolor('black')
                            collection.set_linewidth(1.0)
                        else:
                            collection.set_edgecolor('none')
                            collection.set_linewidth(0)
                        ax.add_collection(collection)
                    except Exception as e:
                         logger.error(f"Error adding patch collection batch: {e}", exc_info=True)
                    patch_collection = []

        if patch_collection:
            if verbose:
                logger.debug(f"Adding final batch of {len(patch_collection)} polygons")
            try:
                collection = matplotlib.collections.PatchCollection(patch_collection, match_original=False)
                collection.set_facecolor(color)
                collection.set_alpha(alpha)
                if show_border:
                    collection.set_edgecolor('black')
                    collection.set_linewidth(1.0)
                else:
                    collection.set_edgecolor('none')
                    collection.set_linewidth(0)
                ax.add_collection(collection)
            except Exception as e:
                 logger.error(f"Error adding final patch collection batch: {e}", exc_info=True)

        if verbose:
            logger.debug(f"Created and added total of {total_faces} polygons for visualization")

        if label:
            ax.plot([], [], color=color, label=label)

    geometry_info: list[dict] = []
    processed_geometries: list[trimesh.Trimesh] = []
    for geom in transformed_geometries:
        if not hasattr(geom, 'vertices') or len(geom.vertices) == 0: continue
        vertices_2d = geom.vertices[:, view_axes_indices]
        vertices_2d = np.dot(vertices_2d, transform_matrix)

        if vertices_2d.shape[0] == 0:
             if verbose: logger.debug("Skipping geometry with no vertices in 2D projection.")
             continue

        min_bounds = np.min(vertices_2d, axis=0)
        max_bounds = np.max(vertices_2d, axis=0)
        center = np.mean([min_bounds, max_bounds], axis=0)
        size = max_bounds - min_bounds

        if size[0] < dimension_threshold or size[1] < dimension_threshold:
            if verbose:
                logger.debug(f"Skipping geometry with small dimensions in view plane: dim1={size[0]:.3f}m, dim2={size[1]:.3f}m")
            continue

        geometry_info.append({
            "bounds": {
                "min": min_bounds,
                "max": max_bounds,
                "center": center
            },
            "dimensions": size,
            "points": vertices_2d
        })
        processed_geometries.append(geom)

    if verbose:
        logger.debug(f"Found {len(geometry_info)} distinct geometries suitable for visualization")
        for i, info in enumerate(geometry_info[:3]):
            logger.debug(f"Geometry {i}:")
            logger.debug(f"  Bounds: dim1=[{info['bounds']['min'][0]:.3f}, {info['bounds']['max'][0]:.3f}], "
                  f"dim2=[{info['bounds']['min'][1]:.3f}, {info['bounds']['max'][1]:.3f}]")
        if len(geometry_info) > 3: logger.debug("  ...")

    return geometry_info, processed_geometries
