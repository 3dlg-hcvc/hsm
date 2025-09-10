import numpy as np
import trimesh
from typing import List, Optional
from hsm_core.utils import get_logger

from .constants import (
    VERTICAL_SEARCH_MARGIN,
    Z_MARGIN,
    EPSILON,
    MIN_SEPARATION,
)

logger = get_logger('support_region.geometry')

def calculate_surface_segments(
    geometries,
    support_data,
    transform_matrix=np.array([[1, 0], [0, 1]]),
    verbose: bool = False,
    min_z_span_ratio: float = 0.8,
    require_separate_geometries: bool = False,
):
    """Calculate surface segments based on vertical surfaces in *support_data*.
    
    Only segments if vertical surfaces actually separate distinct geometries.
    """
    if verbose:
        logger.debug("Starting surface segmentation")

    all_vertices = np.concatenate([geom.vertices[:, [0, 2]] for geom in geometries])
    all_vertices = np.dot(all_vertices, transform_matrix)
    min_bounds = np.min(all_vertices, axis=0)
    max_bounds = np.max(all_vertices, axis=0)
    
    if verbose:
        logger.debug(f"Layer bounds: x=[{min_bounds[0]:.3f}, {max_bounds[0]:.3f}], z=[{min_bounds[1]:.3f}, {max_bounds[1]:.3f}]")
    
    vertical_surfaces = [s for s in support_data['surfaces'] if s['is_vertical']]
    if verbose:
        logger.debug(f"Found {len(vertical_surfaces)} total vertical surfaces")
    
    y_coord = np.mean([geom.vertices[:, 1].mean() for geom in geometries])
    
    if verbose:
        logger.debug(f"Layer y-coordinate: {y_coord:.3f}m")
    
    # Find relevant vertical surfaces that intersect with this layer
    relevant_verticals = []
    for vert in vertical_surfaces:
        # Compute bottom (min-Y) of the vertical surface *without* the search margin.
        vert_bottom_y = vert["centroid"][1] - vert["axes_lengths"][1] / 2

        # Define a narrow band **above** the current layer within which a vertical surface
        # is considered relevant.  This prevents side-faces below the layer from being
        # treated as segmentation dividers.
        vert_min_y = vert_bottom_y - VERTICAL_SEARCH_MARGIN  # bottom of the panel
        vert_max_y = vert_bottom_y + VERTICAL_SEARCH_MARGIN  # small band above the layer

        if verbose:
            logger.debug(
                "Checking vertical surface (strict-above mode): centroid=%s, y_bottom=%.3f, band=[%.3f, %.3f], layer_y=%.3f",
                vert["centroid"], vert_bottom_y, vert_min_y, vert_max_y, y_coord,
            )

        # Only treat the vertical surface as intersecting if the layer lies within the narrow
        # band immediately *below* its bottom edge.  This ensures the panel actually starts
        # on (or just above) the horizontal surface.
        if vert_min_y - EPSILON <= y_coord <= vert_max_y:
            relevant_verticals.append(vert)
            if verbose:
                logger.debug(
                    "Accepted vertical surface at x=%.3f (bottom y=%.3f)",
                    vert["centroid"][0], vert_bottom_y,
                )
        elif verbose:
            logger.debug(
                "Rejected vertical surface at x=%.3f: layer outside band (Δ=%.3f)",
                vert["centroid"][0], y_coord - vert_bottom_y,
            )

    if verbose:
        logger.debug(f"Found {len(relevant_verticals)} intersecting vertical surfaces")

    if not relevant_verticals:
        if verbose:
            logger.debug("No vertical surfaces found - skipping segmentation")
        return None

    # Check each vertical surface to see if it actually separates distinct geometries
    valid_x_positions = set()
    for vert in relevant_verticals:
        half_width = vert['axes_lengths'][0]/2
        vert_center_x = vert['centroid'][0]
        
        z_extent = vert["axes_lengths"][2]
        vert_min_z = vert["centroid"][2] - z_extent / 2
        vert_max_z = vert["centroid"][2] + z_extent / 2

        # Determine whether the surface spans enough of the layer along Z.
        if min_z_span_ratio >= 1.0 - EPSILON:
            # Legacy strict behaviour: must cover full span within a margin.
            spans_enough_z = (vert_min_z <= min_bounds[1] + Z_MARGIN) and (
                vert_max_z >= max_bounds[1] - Z_MARGIN
            )
        else:
            # Fractional coverage accepted.
            layer_z_span = max_bounds[1] - min_bounds[1]
            if layer_z_span <= EPSILON:
                # Degenerate layer, skip.
                if verbose:
                    logger.debug("Layer Z-span too small – skipping span check.")
                spans_enough_z = False
            else:
                intersection_z = max(
                    0.0,
                    min(vert_max_z, max_bounds[1]) - max(vert_min_z, min_bounds[1]),
                )
                z_span_fraction = intersection_z / layer_z_span
                spans_enough_z = z_span_fraction >= min_z_span_ratio
                if verbose:
                    logger.debug(
                        f"Vertical surface at x={vert_center_x:.3f}: Z-span fraction="
                        f"{z_span_fraction:.3f} (threshold={min_z_span_ratio})"
                    )
        
        if not spans_enough_z:
            if verbose:
                logger.debug(f"Vertical surface at x={vert_center_x:.3f} doesn't span enough Z-range")
            continue
        
        # Check if there are separate geometries on both sides
        left_geometries = []
        right_geometries = []
        mid_x = vert_center_x
        
        for geom in geometries:
            verts_2d = np.dot(geom.vertices[:, [0, 2]], transform_matrix)
            geom_min_x = np.min(verts_2d[:, 0])
            geom_max_x = np.max(verts_2d[:, 0])
            
            # Check if geometry is primarily on one side or the other
            if geom_max_x < mid_x - EPSILON:
                left_geometries.append(geom)
            elif geom_min_x > mid_x + EPSILON:
                right_geometries.append(geom)
            # If geometry spans across the vertical surface, don't count it as separated
        
        has_separate_geometries = len(left_geometries) > 0 and len(right_geometries) > 0
        
        if verbose:
            logger.debug(f"Vertical surface at x={vert_center_x:.3f}:")
            logger.debug(f"  Spans enough Z-range: {spans_enough_z}")
            logger.debug(f"  Left geometries: {len(left_geometries)}, Right geometries: {len(right_geometries)}")
            logger.debug(f"  Separates geometries: {has_separate_geometries} (required={require_separate_geometries})")
        
        if (not require_separate_geometries) or has_separate_geometries:
            x_min = round(vert_center_x - half_width, 3)
            x_max = round(vert_center_x + half_width, 3)
            valid_x_positions.add(x_min)
            valid_x_positions.add(x_max)
            if verbose:
                logger.debug(f"  Adding segment boundaries: x=[{x_min:.3f}, {x_max:.3f}]")
        elif verbose:
            logger.debug("  Skipping vertical surface: does not separate distinct geometries")
    
    # Add layer boundaries
    valid_x_positions.add(round(min_bounds[0], 3))
    valid_x_positions.add(round(max_bounds[0], 3))
    
    if verbose:
        logger.debug(f"Raw x-positions: {[f'{x:.3f}' for x in sorted(list(valid_x_positions))]}")
    
    # Remove duplicate positions that are too close
    unique_x = []
    if valid_x_positions:
        sorted_x = sorted(list(valid_x_positions))
        unique_x.append(sorted_x[0])
        for x in sorted_x[1:]:
            if abs(x - unique_x[-1]) > MIN_SEPARATION:
                unique_x.append(round(x, 3))
            else:
                # Average with the last one if they are too close
                unique_x[-1] = round((unique_x[-1] + x) / 2, 3)

    logger.info(f"Final segmentation x-positions: {[f'{x:.3f}' for x in unique_x]}")
    return unique_x if len(unique_x) > 2 else None  # Only return if we have actual segments

def segment_geometry(geometry: trimesh.Trimesh, x_min: float, x_max: float, verbose: bool = False) -> Optional[trimesh.Trimesh]:
    """Extract a segment of geometry between x_min and x_max."""
    if verbose:
        logger.debug(f"Segmenting geometry between x=[{x_min:.3f}, {x_max:.3f}]")
        logger.debug(f"Input geometry has {len(geometry.faces)} faces")
    
    vertices = geometry.vertices
    eps = 1e-6
    x_min_b, x_max_b = x_min - eps, x_max + eps
    
    face_vertices = vertices[geometry.faces]
    face_x_coords = face_vertices[:, :, 0]
    
    face_mins = np.min(face_x_coords, axis=1)
    face_maxs = np.max(face_x_coords, axis=1)
    valid_faces_mask = (
        ((face_x_coords >= x_min_b) & (face_x_coords <= x_max_b)).any(axis=1) |
        ((face_mins <= x_min_b) & (face_maxs >= x_max_b))
    )
    
    if not np.any(valid_faces_mask):
        return None
    
    valid_faces = geometry.faces[valid_faces_mask]
    
    used_vertices, inverse = np.unique(valid_faces.flatten(), return_inverse=True)
    new_faces = inverse.reshape(-1, 3)
    new_vertices = vertices[used_vertices]
    
    new_vertices[:, 0] = np.clip(new_vertices[:, 0], x_min, x_max)
    
    result = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    result.remove_unreferenced_vertices()
    
    if verbose:
        logger.debug(f"Segmented mesh: {len(result.vertices)} vertices, {len(result.faces)} faces")
    
    return result

def merge_touching_geometries(
    geometries: List[trimesh.Trimesh], 
    merge_distance: float = 0.02, 
    verbose: bool = False
) -> List[trimesh.Trimesh]:
    """Merge geometries that are within a specified distance of each other."""
    if len(geometries) <= 1:
        return geometries
        
    if verbose:
        logger.debug(f"Attempting to merge {len(geometries)} geometries with merge distance: {merge_distance}m")
    
    result = [geom.copy() for geom in geometries]
    active = [True] * len(result)
    
    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(len(result)):
            if not active[i]:
                continue
            for j in range(i + 1, len(result)):
                if not active[j]:
                    continue
                try:
                    verts1_2d = result[i].vertices[:, [0, 2]]
                    verts2_2d = result[j].vertices[:, [0, 2]]
                    distances = np.linalg.norm(verts1_2d[:, None] - verts2_2d, axis=2)
                    min_distance = np.min(distances)
                    
                    if min_distance <= merge_distance:
                        if verbose:
                            logger.debug(f"Merging geometries {i} and {j} (distance: {min_distance:.4f}m)")
                        result[i] = trimesh.util.concatenate([result[i], result[j]])
                        active[j] = False
                        merged_any = True
                        break
                except Exception as e:
                    if verbose:
                        logger.warning(f"Error calculating distance between geometries {i} and {j}: {e}")
            if merged_any:
                break
    
    final_result = [result[i] for i in range(len(result)) if active[i]]
    if verbose:
        logger.debug(f"Reduced from {len(geometries)} to {len(final_result)} geometries")
    
    return final_result 