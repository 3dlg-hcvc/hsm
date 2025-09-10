from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches
import trimesh
from typing import Dict, Any, List, Tuple, Optional
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from hsm_core.utils import get_logger

from .constants import (
    NORMAL_HORIZONTAL_THRESHOLD, AREA_THRESHOLD, SIZE_THRESHOLD, MERGE_DISTANCE,
    HEIGHT_THRESHOLD_FACTOR, COVERAGE_RATIO_THRESHOLD, MIN_AVAILABLE_SPACE_ABOVE_LAYER,
    PADDING, SURFACE_COLORS, MIN_AREA, TOP_HEIGHT
)
from .loader import parse_support_surface
from .geometry import calculate_surface_segments, merge_touching_geometries

from .utils import build_surface_data_entry, round_nested_dict, simplify_vertices_outer

logger = get_logger('support_region.analyzer')

def _get_valid_heights(parsed_data):
    exact_heights = set()
    for surface in parsed_data['surfaces']:
        if abs(surface['normal'][1]) > NORMAL_HORIZONTAL_THRESHOLD:
            height = round(surface['centroid'][1], 4)
            exact_heights.add(height)
        else:
            logger.debug(f"Surface {surface['normal'][1]} is not horizontal")
    return sorted(list(exact_heights), reverse=True)

def _group_geometries_by_height(scene, valid_heights):
    if not valid_heights:
        return {}
    geometries_by_height = {}
    
    for name, geometry in scene.geometry.items():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        height = np.min(geometry.vertices[:, 1])
        closest_height = min(valid_heights, key=lambda x: abs(x - height))
        if closest_height not in geometries_by_height:
            geometries_by_height[closest_height] = []
        geometries_by_height[closest_height].append(geometry)
    return {h: g for h, g in geometries_by_height.items() if h in valid_heights}

def _filter_geometries(geometries_by_height, verbose=False):
    filtered_geometries_by_height = {}
    for height, geometries in geometries_by_height.items():
        valid_geometries = []
        if verbose:
            logger.debug(f"Analyzing height {height:.3f}m:")
        for i, geom in enumerate(geometries):
            area = geom.area
            if verbose:
                logger.debug(f"  Geometry {i}: area={area:.3f}m²")
            if area > AREA_THRESHOLD:
                bounds = geom.bounds
                x_extent = bounds[1][0] - bounds[0][0]
                z_extent = bounds[1][2] - bounds[0][2]
                if x_extent < SIZE_THRESHOLD or z_extent < SIZE_THRESHOLD:
                    if verbose:
                        logger.debug(f"    geometry too thin -> x_extent: {x_extent:.3f}m, z_extent: {z_extent:.3f}m")
                else:
                    valid_geometries.append(geom)
            else:
                if verbose:
                    logger.debug(f"    Filtered out: area {area:.3f}m² below threshold {AREA_THRESHOLD:.3f}m²")
        
        if valid_geometries:
            filtered_geometries_by_height[height] = valid_geometries
            if verbose and len(valid_geometries) != len(geometries):
                logger.debug(f"Layer at height {height:.3f}m: Filtered out {len(geometries) - len(valid_geometries)} geometries")
        elif verbose:
            logger.debug(f"  No valid geometries at height {height:.3f}m - entire layer filtered out")
    return filtered_geometries_by_height

def _calculate_layer_heights_and_filter(valid_heights, verbose=False):
    sorted_heights = sorted(valid_heights)
    layer_heights = {sorted_heights[i]: sorted_heights[i + 1] - sorted_heights[i] for i in range(len(sorted_heights) - 1)}
    
    temp_filtered_heights = []
    for height_val in valid_heights:
        space_above = layer_heights.get(height_val, TOP_HEIGHT)  # Use TOP_HEIGHT for top surfaces
        if space_above >= MIN_AVAILABLE_SPACE_ABOVE_LAYER:
            temp_filtered_heights.append(height_val)
        elif verbose:
            logger.debug(f"Filtering out layer at {height_val:.3f}m due to insufficient space above: {space_above:.3f}m")
    return temp_filtered_heights, layer_heights

def _detect_top_layers(geometries_by_height, valid_heights, verbose=False):
    layer_info = []
    for height in valid_heights:
        geometries = geometries_by_height[height]
        layer_bounds = np.vstack([g.bounds for g in geometries])
        min_bounds, max_bounds = np.min(layer_bounds, axis=0), np.max(layer_bounds, axis=0)
        
        total_faces = sum(len(g.face_normals) for g in geometries)
        upward_faces = sum(np.sum(g.face_normals[:, 1] > 0.9) for g in geometries)
        
        layer_info.append({
            'height': height, 'min_bounds': min_bounds, 'max_bounds': max_bounds,
            'total_faces': total_faces, 'upward_faces': upward_faces
        })

    top_layers = []
    if not layer_info: return [], []
    
    total_height = max(l['height'] for l in layer_info) if layer_info else 0
    height_threshold = total_height * HEIGHT_THRESHOLD_FACTOR
    
    if verbose:
        logger.debug("\nTop Layer Detection Analysis")
        logger.debug(f"Total height: {total_height:.3f}m, Height threshold: {height_threshold:.3f}m")

    for i, layer in enumerate(layer_info):
        if verbose:
            logger.debug(f"\nAnalyzing layer at height {layer['height']:.3f}m:")
            logger.debug(f"  - Above height threshold: {layer['height'] >= height_threshold}")
            if layer['total_faces'] > 0:
                logger.debug(f"  - Upward faces ratio: {layer['upward_faces']}/{layer['total_faces']} = {layer['upward_faces']/layer['total_faces']:.2f}")

        if layer['height'] < height_threshold:
            if verbose: logger.debug("  → Skipped: Below height threshold")
            continue
        
        is_covered = False
        for higher_layer in layer_info[:i]:
            if higher_layer['height'] <= layer['height']: continue
            
            overlap_x = min(higher_layer['max_bounds'][0], layer['max_bounds'][0]) - max(higher_layer['min_bounds'][0], layer['min_bounds'][0])
            overlap_z = min(higher_layer['max_bounds'][2], layer['max_bounds'][2]) - max(higher_layer['min_bounds'][2], layer['min_bounds'][2])
            
            if overlap_x > 0 and overlap_z > 0:
                layer_area = (layer['max_bounds'][0] - layer['min_bounds'][0]) * (layer['max_bounds'][2] - layer['min_bounds'][2])
                if layer_area > 1e-6:
                    coverage_ratio = (overlap_x * overlap_z) / layer_area
                    if verbose: logger.debug(f"  - Coverage by layer at {higher_layer['height']:.3f}m: {coverage_ratio:.2%}")
                    if coverage_ratio > COVERAGE_RATIO_THRESHOLD:
                        is_covered = True
                        if verbose: logger.debug(f"  → Covered by layer at {higher_layer['height']:.3f}m")
                        break
        
        if not is_covered:
            top_layers.append(layer['height'])
            if verbose: logger.debug("  → Marked as top layer")

    return top_layers, layer_info

def _calculate_global_bounds(geometries_by_height, valid_heights, transform_matrix_np):
    """Calculate global bounds for relative scaling."""
    global_min_x = float('inf')
    global_max_x = float('-inf')
    global_min_z = float('inf')
    global_max_z = float('-inf')
    
    for height in valid_heights:
        for geometry in geometries_by_height[height]:
            vertices_2d = geometry.vertices[:, [0, 2]]
            vertices_2d = np.dot(vertices_2d, transform_matrix_np)
            min_bounds = np.min(vertices_2d, axis=0)
            max_bounds = np.max(vertices_2d, axis=0)
            
            global_min_x = min(global_min_x, min_bounds[0])
            global_max_x = max(global_max_x, max_bounds[0])
            global_min_z = min(global_min_z, min_bounds[1])
            global_max_z = max(global_max_z, max_bounds[1])
    
    padding = PADDING * max(global_max_x - global_min_x, global_max_z - global_min_z)
    global_min_x -= padding
    global_max_x += padding
    global_min_z -= padding
    global_max_z += padding
    
    return global_min_x, global_max_x, global_min_z, global_max_z

def _setup_visualization(valid_heights, visualize):
    """Setup matplotlib figure and axes for visualization."""
    fig, axes = None, []
    if visualize:
        max_cols = 2
        n_layers = len(valid_heights)
        if n_layers > 0:
            n_rows = (n_layers + max_cols - 1) // max_cols
            n_cols = min(n_layers, max_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
            axes = axes.flatten()
            for idx in range(n_layers, len(axes)):
                axes[idx].set_visible(False)
    return fig, axes

def _create_2d_polygons(geometries, transform_matrix_np, verbose=False):
    """Create 2D polygons from 3D geometries."""
    list_of_2d_polygons = []
    for geom in geometries:
        vertices_2d = np.dot(geom.vertices[:, [0, 2]], transform_matrix_np)
        outer_data = simplify_vertices_outer(vertices_2d, faces=geom.faces, return_holes=True)
        # The helper returns (shell, holes) when `return_holes` is True.
        if isinstance(outer_data, tuple):
            shell, holes = outer_data
        else:
            shell, holes = outer_data, []

        if shell.shape[0] >= 3:
            try:
                list_of_2d_polygons.append(Polygon(shell, holes))
            except Exception as e:
                if verbose:
                    logger.debug(f"Failed to build Polygon with holes: {e}. Falling back to exterior only.")
                    list_of_2d_polygons.append(Polygon(shell))
    return list_of_2d_polygons

def _merge_geometries_2d(geometries, transform_matrix_np, verbose=False):
    """Create 2D polygons from geometries and optionally merge them."""
    logger.debug(f"Processing geometries")
    
    # Create individual 2D polygons
    list_of_2d_polygons = _create_2d_polygons(geometries, transform_matrix_np, verbose)
    
    # Return individual polygons
    final_polygons = []
    if list_of_2d_polygons:
        # if merge:
        #     logger.debug("Merging 2D polygons using unary_union")
        #     merged_geometry = unary_union(list_of_2d_polygons)
            
        #     if isinstance(merged_geometry, Polygon):
        #         final_polygons.append(merged_geometry)
        #     elif isinstance(merged_geometry, MultiPolygon):
        #         final_polygons.extend(merged_geometry.geoms)
        # else:
        logger.debug("Returning individual polygons without merging")
        final_polygons = list_of_2d_polygons
    
    return final_polygons

def _visualize_and_create_surface_data(final_polygons, ax, visualize, color_index, height, layer_bounds):
    """Visualize polygons and create surface data entries."""
    surface_info = []
    surface_id = 0
    
    for poly in final_polygons:
        if not poly.is_valid or poly.area < MIN_AREA:
            continue

        color_name, color_hex = SURFACE_COLORS[color_index % len(SURFACE_COLORS)]
        
        if visualize and ax:
            patch = matplotlib.patches.Polygon(
                np.array(poly.exterior.coords), closed=True,
                facecolor=color_hex, alpha=1.0, fill=True,
                edgecolor='black', linewidth=1.0
            )
            ax.add_patch(patch)
        
        # Build info_item from polygon properties
        min_x, min_y, max_x, max_y = poly.bounds
        bounds_min = np.array([min_x, min_y])
        bounds_max = np.array([max_x, max_y])
        
        info_item = {
            "bounds": {"min": bounds_min, "max": bounds_max, "center": (bounds_min + bounds_max) / 2},
            "dimensions": bounds_max - bounds_min,
            "points": np.array(poly.exterior.coords)
        }

        surface_info.append(build_surface_data_entry(
            info_item, None, surface_id, color_name, height, layer_bounds,
            'relative_to_layer', simplified_vertices_override=np.array(poly.exterior.coords).tolist()
        ))
        surface_id += 1
        color_index += 1
    
    return surface_info, color_index

def _process_segmented_surfaces(segments, geometries, ax, visualize, verbose, transform_matrix_np,
                               color_index, height, layer_bounds):
    """Process surfaces that have been segmented by vertical elements."""
    from hsm_core.visualization.geometry_2d_visualizer import visualize_object_geometry
    surface_info = []
    if segments is not None:
        surface_id = 0
        for i in range(len(segments) - 1):
            x_min, x_max = segments[i], segments[i+1]
            color_name, color_hex = SURFACE_COLORS[color_index % len(SURFACE_COLORS)]
            
            geo_info, proc_geoms = visualize_object_geometry(
                geometries, ax, (color_hex if visualize else None), verbose=verbose, 
                x_bounds=(x_min, x_max), transform_matrix=transform_matrix_np
            )
            
            if geo_info:
                for info, geom in zip(geo_info, proc_geoms):
                    surface_info.append(build_surface_data_entry(
                        info, geom, surface_id, color_name, height, layer_bounds,
                        'relative_to_layer', width_override_val=(x_max - x_min),
                        bounds_override_val={'min': [x_min, info['bounds']['min'][1]], 'max': [x_max, info['bounds']['max'][1]]}
                    ))
                    surface_id += 1
                color_index += 1

    # De-duplicate identical surfaces
    if surface_info:
        unique = []
        seen_keys = set()
        for s in surface_info:
            key = (
                round(height, 2),
                round(s["bounds"]["min"][0], 2),
                round(s["bounds"]["min"][1], 2),
                round(s["bounds"]["max"][0], 2),
                round(s["bounds"]["max"][1], 2),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(s)
        surface_info = unique

    return surface_info, color_index

def _configure_axis(ax, relative_scale, global_bounds):
    """Configure axis properties for visualization."""
    if ax:
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Z (meters)')
        
        if relative_scale and global_bounds[0] is not None:
            global_min_x, global_max_x, global_min_z, global_max_z = global_bounds
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_z, global_max_z)
        else:
            ax.autoscale_view()

def _finalize_visualization(fig, visualize, output_path):
    """Finalize and save visualization if needed."""
    if visualize and fig:
        plt.tight_layout()
        if output_path:
            save_path = Path(output_path) / "support_surface_visualization.png"
            plt.savefig(save_path)
            logger.info(f"Support surface visualization saved to: {save_path}")

def extract_support_region(
    scene: trimesh.Scene, 
    support_data: Dict[str, Any], 
    verbose: bool = False, 
    relative_scale: bool = True, 
    output_path: Optional[str] = None, 
    skip_segmentation: bool = False, 
    shrink_factor: float = 1.0, 
    transform_matrix: List[List[float]] = [[-1, 0], [0, 1]],
    visualize: bool = True, 
    merge: bool = True
) -> Tuple[Optional[Figure], Dict[str, Any]]:
    """Extract support region from preprocessed support surface data."""
    
    transform_matrix_np = np.array(transform_matrix)
    parsed_data = parse_support_surface(support_data)
    
    # Pipeline
    valid_heights = _get_valid_heights(parsed_data)
    
    if not valid_heights:
        logger.warning("No valid heights found in support data.")
        return None, {}
    
    geometries_by_height = _group_geometries_by_height(scene, valid_heights)
    geometries_by_height = _filter_geometries(geometries_by_height, verbose)
    
    valid_heights = sorted(geometries_by_height.keys(), reverse=True)
    
    if not valid_heights:
        logger.warning("No valid layers found after filtering geometries.")
        return None, {}
    
    valid_heights, layer_heights = _calculate_layer_heights_and_filter(valid_heights, verbose)
    top_layers, layer_info = _detect_top_layers(geometries_by_height, valid_heights, verbose)

    if not valid_heights:
        logger.warning("No valid layers found for visualization.")
        return None, {}

    # Calculate global bounds for relative scaling if needed
    global_bounds = None
    if relative_scale and visualize:
        global_bounds = _calculate_global_bounds(geometries_by_height, valid_heights, transform_matrix_np)

    # Visualization setup
    fig, axes = _setup_visualization(valid_heights, visualize)
    
    layer_data = {}
    color_index = 0

    for idx, height in enumerate(valid_heights):
        ax = axes[idx] if visualize else None
        logger.info(f"Processing layer at height {height:.3f}m")
        geometries = geometries_by_height[height]
        is_top_layer = height in top_layers

        if visualize and ax:
            title = f'Layer {idx}\n(y={height:.2f}m)'
            if is_top_layer: title += ' (Top)'
            if height in layer_heights: title += f'\nSpace: {layer_heights[height]:.2f}m'
            ax.set_title(title)

        segments = None
        if not is_top_layer and not skip_segmentation:
            segments = calculate_surface_segments(geometries, parsed_data, verbose=verbose, transform_matrix=transform_matrix_np)

        all_vertices = np.concatenate([geom.vertices[:, [0, 2]] for geom in geometries])
        all_vertices = np.dot(all_vertices, transform_matrix_np)
        layer_bounds = {'min': np.min(all_vertices, axis=0), 'max': np.max(all_vertices, axis=0)}
        
        surface_info = []
        process_as_single_unit = is_top_layer or skip_segmentation or segments is None or len(segments) <= 1

        if process_as_single_unit:
            final_polygons = _merge_geometries_2d(geometries, transform_matrix_np, verbose)
            surface_info, color_index = _visualize_and_create_surface_data(
                final_polygons, ax, visualize, color_index, height, layer_bounds
            )
        else: 
            # Process segments
            surface_info, color_index = _process_segmented_surfaces(
                segments, geometries, ax, visualize, verbose, transform_matrix_np, 
                color_index, height, layer_bounds
            )

        _configure_axis(ax, relative_scale, global_bounds)
        
        layer_data[f"layer_{idx}"] = {
            'height': height, 'is_top_layer': is_top_layer,
            'space_above': layer_heights.get(height, TOP_HEIGHT), 'surfaces': surface_info
        }
        if verbose:
            logger.debug(f"\nLayer {idx}: {'Top surface' if is_top_layer else ''}")
            logger.debug(f"  Height: {height:.3f}m, Surfaces found: {len(surface_info)}")

    _finalize_visualization(fig, visualize, output_path)
    
    return fig, round_nested_dict(layer_data) 