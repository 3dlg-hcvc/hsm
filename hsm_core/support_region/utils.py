from collections import Counter, defaultdict
import numpy as np
from shapely import MultiPolygon, Polygon
import trimesh
from pathlib import Path as PathLib
from trimesh.caching import TrackedArray
import time
from hsm_core.utils import get_logger

from .constants import VERTEX_MERGE_THRESHOLD

logger = get_logger('support_region.utils')

def build_surface_data_entry(info_item, geom_item, surface_id_val, color_val, height_val,
                             layer_bounds_for_relative_center_val,
                             center_mode_val='absolute',
                             width_override_val=None, bounds_override_val=None,
                             simplified_vertices_override=None):
    """Helper function to build a single surface data entry.
    Simplify vertices happen here."""
    # If pre-computed vertices are provided, use them. Otherwise, compute them.
    if simplified_vertices_override is not None:
        simplified_vertices = simplified_vertices_override
    else:
        simplified_vertices = simplify_vertices_outer(info_item['points'], faces=geom_item.faces).tolist()

    actual_width = width_override_val if width_override_val is not None else info_item['dimensions'][0]
    depth = info_item['dimensions'][1]  # Consistent across uses
    actual_area = actual_width * depth

    if bounds_override_val:
        actual_bounds_min = bounds_override_val['min']
        actual_bounds_max = bounds_override_val['max']
    else:
        actual_bounds_min = info_item['bounds']['min'].tolist()
        actual_bounds_max = info_item['bounds']['max'].tolist()

    surface_center_in_layer_xz_plane = info_item['bounds']['center']

    if center_mode_val == 'relative_to_layer':
        if layer_bounds_for_relative_center_val is None:
            raise ValueError("layer_bounds_for_relative_center_val is required for relative center mode")
        layer_center_xz_plane = np.mean([layer_bounds_for_relative_center_val['min'], layer_bounds_for_relative_center_val['max']], axis=0)
        display_center = (surface_center_in_layer_xz_plane - layer_center_xz_plane).tolist()
    elif center_mode_val == 'absolute':
        display_center = surface_center_in_layer_xz_plane.tolist()
    else:
        raise ValueError(f"Invalid center_mode_val: {center_mode_val}")

    local_transform_matrix = np.eye(4)
    local_transform_matrix[0, 3] = surface_center_in_layer_xz_plane[0]
    local_transform_matrix[1, 3] = height_val
    local_transform_matrix[2, 3] = surface_center_in_layer_xz_plane[1]

    return {
        'surface_id': surface_id_val,
        'color': color_val,
        'width': actual_width,
        'depth': depth,
        'area': actual_area,
        'center': display_center,
        'bounds': {
            'min': actual_bounds_min,
            'max': actual_bounds_max
        },
        'geometry': {
            'vertices': simplified_vertices
        },
        'local_transform': local_transform_matrix.tolist()
    }

def ensure_output_directory(output_path):
    """Create output directory if it doesn't exist"""
    PathLib(output_path).mkdir(parents=True, exist_ok=True)
    return output_path

def log_scene_graph(scene): 
    logger.info("\nScene Graph Information:")
    logger.info("------------------------")
    logger.info(f"Scene type: {type(scene)}")
    logger.info(f"Number of geometries: {len(scene.geometry)}")
    logger.info("\nGeometry details:")
    
    logger.info(scene.geometry.keys())
    logger.info("="*100)
    for name, geom in scene.geometry.items():
        logger.info(f"\nGeometry '{name}':")
        logger.info(f"  Type: {type(geom)}")
        
        if isinstance(geom, trimesh.Trimesh):
            logger.info(f"  Vertices: {len(geom.vertices)}")
            logger.info(f"  Faces: {len(geom.faces)}")
            logger.info(f"  Area: {geom.area:.3f}")
            logger.info(f"  Bounds: {geom.bounds}")
            
            if hasattr(geom, 'metadata') and geom.metadata:
                logger.info("  Metadata:")
                for key, value in geom.metadata.items():
                    logger.info(f"    {key}: {value}")

def shrink_bounds(min_bounds, max_bounds, shrink_factor):
    """Shrink bounds by a factor while maintaining center point."""
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    new_size = size * shrink_factor
    new_min = center - new_size / 2
    new_max = center + new_size / 2
    return new_min, new_max

def simplify_vertices_outer(
    vertices_2d: np.ndarray,
    faces: TrackedArray,
    atol: float = VERTEX_MERGE_THRESHOLD,
    timeout_seconds: float = 10.0,
    return_holes: bool = False,
):
    """   
    Simplify 2D vertices by finding the outer boundary with timeout protection.
    
    Args:
        vertices_2d: 2D vertex array
        faces: Face array
        atol: Tolerance for vertex merging
        timeout_seconds: Maximum time allowed for processing
        return_holes: Whether to return interior rings (holes)
        
    Returns:
        Simplified vertices array, and holes if return_holes is True
    """
    start_time = time.time()
    
    # Early path: if the caller needs interior rings, rely on Shapely's
    # `unary_union` to build a valid polygon and return the shell + holes
    # immediately.  This avoids the more expensive custom edge-tracing below.
    if return_holes:
        try:
            from shapely.geometry import Polygon as _Poly
            from shapely.ops import unary_union

            tri_polys = [_Poly(vertices_2d[face]) for face in faces]
            merged = unary_union(tri_polys)

            if merged.is_empty:
                return np.array([]), []

            # Keep the largest connected component if several exist.
            if isinstance(merged, MultiPolygon):
                merged = max(merged.geoms, key=lambda p: p.area)

            shell = np.array(merged.exterior.coords)[:-1]
            holes = [np.array(ring.coords)[:-1] for ring in merged.interiors]

            return shell, holes
        except Exception as e:
            logger.warning(
                f"Shapely union failed while extracting holes ({e}); falling back to outline-only path.")

    # early return for degenerate inputs
    if len(vertices_2d) < 3:
        if len(vertices_2d) == 0:
            return np.array([])
        return vertices_2d

    overlapping_vertices = {}
    unique_vertices = []
    unique_map = {}

    for i, vertex in enumerate(vertices_2d):
        is_unique = True
        for j, other_vertex in enumerate(unique_vertices):
            if np.allclose(vertex, other_vertex, atol=atol):
                overlapping_vertices[i] = unique_map[j]
                is_unique = False
                break
        if is_unique:
            unique_map[len(unique_vertices)] = i
            overlapping_vertices[i] = i
            unique_vertices.append(vertex)

    edge_count = Counter()
    for face in faces:
        v_indices = {overlapping_vertices[face[0]], overlapping_vertices[face[1]], overlapping_vertices[face[2]]}
        if len(v_indices) < 3:
            continue
        
        for edge in map(
            frozenset,
            [
                {overlapping_vertices[face[0]], overlapping_vertices[face[1]]},
                {overlapping_vertices[face[1]], overlapping_vertices[face[2]]},
                {overlapping_vertices[face[2]], overlapping_vertices[face[0]]},
            ],
        ):
            edge_count[tuple(sorted(edge))] += 1
    outer_edges = [edge for edge, count in edge_count.items() if count == 1 and len(edge) == 2]
    
    if not outer_edges:
        hull_polygon = Polygon(vertices_2d).convex_hull
        if isinstance(hull_polygon, Polygon):
            return np.array(hull_polygon.exterior.coords)[:-1]
        else:
            return np.array([vertices_2d[i] for i in sorted(unique_map.values())])

    edge_mapping = defaultdict(list)
    for edge in outer_edges:
        start, end = list(edge)
        edge_mapping[start].append(end)
        edge_mapping[end].append(start)

    edge_indices = [outer_edges[0][0]]
    visited_edges = set()  # Track visited edges to detect cycles
    max_iterations = len(outer_edges) * 2  # Safety limit based on edge count
    iteration_count = 0
    
    while (len(edge_indices) <= 1 or edge_indices[0] != edge_indices[-1]) and iteration_count < max_iterations:
        # Check timeout periodically
        if iteration_count % 100 == 0 and time.time() - start_time > timeout_seconds:
            logger.warning("Timeout in edge tracing loop: falling back to convex hull")
            hull_polygon = Polygon(vertices_2d).convex_hull
            if isinstance(hull_polygon, Polygon):
                return np.array(hull_polygon.exterior.coords)[:-1]
            else:
                return vertices_2d
        
        current_vertex = edge_indices[-1]
        new_vertices = edge_mapping[current_vertex]
        
        if len(edge_indices) == 1:
            next_vertex = new_vertices[0]
        else:
            # Choose next vertex that's not the previous one
            prev_vertex = edge_indices[-2]
            candidates = [v for v in new_vertices if v != prev_vertex]
            if not candidates:
                # If no valid candidates, we might be stuck
                logger.warning("No valid candidates in edge tracing: falling back to convex hull")
                break
            next_vertex = candidates[0]
        
        # Check for cycle detection (visiting same edge twice)
        edge_key = tuple(sorted([current_vertex, next_vertex]))
        if edge_key in visited_edges:
            logger.warning("Cycle detected in edge tracing: falling back to convex hull")
            break
        visited_edges.add(edge_key)
        
        edge_indices.append(next_vertex)
        iteration_count += 1

    # If we successfully traced a closed loop
    if len(edge_indices) > 3 and edge_indices[0] == edge_indices[-1]:
        logger.debug(f"edge_indices: {edge_indices}")
        final_vertices = vertices_2d[edge_indices[:-1]]
        traced_shape = Polygon(final_vertices)
        
        if traced_shape.is_valid:
            return final_vertices
        else:
            logger.warning(f"traced_shape is invalid: {traced_shape}")
            repaired_shape = traced_shape.buffer(0)
            if repaired_shape.is_valid:
                logger.info("Successfully repaired invalid polygon using .buffer(0)")
                if isinstance(repaired_shape, MultiPolygon):
                    repaired_shape = max(repaired_shape.geoms, key=lambda p: p.area)
                
                if isinstance(repaired_shape, Polygon):
                     return np.array(repaired_shape.exterior.coords)[:-1]

    # Fallback to convex hull if edge tracing failed
    logger.warning("Edge tracing failed or timed out, using convex hull fallback")
    hull_polygon = Polygon(vertices_2d).convex_hull
    if isinstance(hull_polygon, Polygon):
        return np.array(hull_polygon.exterior.coords)[:-1]
    else:
        return np.array([vertices_2d[i] for i in sorted(unique_map.values())])

def round_nested_dict(obj):
    decimals = 4
    if isinstance(obj, dict):
        return {k: round_nested_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_nested_dict(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return np.round(obj, decimals=decimals)
    elif isinstance(obj, float):
        return np.round(obj, decimals=decimals)
    else:
        return obj 