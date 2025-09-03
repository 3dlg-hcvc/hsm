"""Scene object placement."""

import sys
import logging
from shapely.geometry import Polygon, LineString, Point
import trimesh
import numpy as np
from pathlib import Path
from typing import List, Dict, Literal, Optional, Tuple
import shapely.affinity
import os
import logging

from hsm_core.scene.validate import *
from hsm_core.scene.objects import SceneObject
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.motif import SceneMotif
from hsm_core.scene.spec import ObjectSpec
from hsm_core.constants import *

from hsm_core.scene_motif.core.arrangement import Arrangement

WORLD_UP: np.ndarray = np.array([0, 1, 0])
WORLD_FORWARD: np.ndarray = np.array([0, 0, 1])

MIN_CUTOUT_CORNER_MARGIN: float = 0.1  # Minimum distance from corner for cutouts
MIN_CUTOUT_SEPARATION: float = 0.2  # Minimum separation between cutouts
MIN_WINDOW_DOOR_SEPARATION: float = 0.1  # Reduced separation between windows and doors

WALL_CORNER_MARGIN: float = 0.2  # Margin from wall corners
CUTOUT_WALL_DISTANCE_THRESHOLD: float = 0.1  # Maximum distance from wall for cutout to be valid
ARROW_SHAFT_PROPORTION: float = 0.7  # Proportion of arrow that is shaft
ARROW_HEAD_PROPORTION: float = 0.3  # Proportion of arrow that is head

logger = logging.getLogger(__name__)

class Cutout:
    """
    Class to represent and manage wall cutouts (doors and windows).
    """
    CUTOUT_TYPE = Literal["door", "window"]
    
    def __init__(self, 
                 location: Tuple[float, float], 
                 cutout_type: CUTOUT_TYPE = "door", 
                 width: float = 0.0, 
                 height: float = 0.0,
                 bottom_height: float = 0.0):
        """
        Initialize a cutout with position and dimensions.
        
        Args:
            location: (x, y) position of the cutout's center
            cutout_type: "door" or "window"
            width: Width of the cutout in meters (if None, uses default for type)
            height: Height of the cutout in meters (if None, uses default for type)
            bottom_height: Height from floor to bottom of cutout (if None, uses default for type)
        """
        self.location = location
        self.cutout_type = cutout_type.lower()
        self.original_location = location  # Store original location for reference
        
        # Set defaults based on type if not specified
        if self.cutout_type == "door":
            self.width = width if width > 0 else DOOR_WIDTH
            self.height = height if height > 0 else DOOR_HEIGHT
            self.bottom_height = bottom_height if bottom_height > 0 else 0.0
        elif self.cutout_type == "window":
            self.width = width if width > 0 else WINDOW_WIDTH
            self.height = height if height > 0 else WINDOW_HEIGHT
            self.bottom_height = bottom_height if bottom_height > 0 else WINDOW_BOTTOM_HEIGHT
            self.original_width = self.width  # Store original width for reference
        else:
            raise ValueError(f"Invalid cutout type: {cutout_type}")
            
        # Validate location
        self.is_valid = True
        self.closest_wall_index = -1
        self.distance_to_wall = float('inf')
        
        # Store wall projection data for overlap checking
        self.wall_start = None
        self.wall_end = None
        self.projection_on_wall: float = 0.0
        self.wall_length: float = 0.0
    
    def __str__(self) -> str:
        """String representation of the cutout."""
        return f"{self.cutout_type.capitalize()} at {self.location}, size: {self.width}x{self.height}m, height from floor: {self.bottom_height}m"
    
    def validate(self, room_polygon: Polygon, existing_cutouts: List['Cutout'] = []) -> bool:
        """
        Validate that the cutout location is properly positioned within the room
        and doesn't overlap with existing cutouts.
        
        Args:
            room_polygon: Shapely Polygon representing the room
            existing_cutouts: List of already placed cutouts to check for overlaps
            
        Returns:
            bool: True if the cutout is valid, False otherwise
        """
        # First check if the point is inside or on the boundary of the room
        point = Point(self.location)
        if not (room_polygon.contains(point) or room_polygon.boundary.contains(point)):
            print(f"Warning: {self.cutout_type} at {self.location} is outside the room")
            self.is_valid = False
            return False
            
        # Find closest wall and calculate distance
        wall_segments = list(zip(room_polygon.exterior.coords[:-1], room_polygon.exterior.coords[1:]))
        min_distance = float('inf')
        closest_wall_index = -1
        
        for i, (start, end) in enumerate(wall_segments):
            wall_line = LineString([start, end])
            distance = wall_line.distance(point)
            
            if distance < min_distance:
                min_distance = distance
                closest_wall_index = i
                self.wall_start = start
                self.wall_end = end
                
        self.closest_wall_index = closest_wall_index
        self.distance_to_wall = min_distance
        
        # Ensure cutout is close to a wall
        if min_distance > CUTOUT_WALL_DISTANCE_THRESHOLD:
            print(f"Warning: {self.cutout_type} at {self.location} is too far from any wall ({min_distance:.2f}m)")
            self.is_valid = False
            return False

        # Check wall length sufficiency
        wall_start, wall_end = wall_segments[closest_wall_index]
        wall_length = LineString([wall_start, wall_end]).length
        self.wall_length = wall_length
        
        if wall_length < self.width + WALL_CORNER_MARGIN * 2:  # Add margin on both sides
            print(f"Warning: Wall is too short ({wall_length:.2f}m) for {self.cutout_type} of width {self.width}m")
            
            # For windows, try to reduce width to fit
            if self.cutout_type == "window":
                max_width = wall_length - WALL_CORNER_MARGIN * 2  # Leave margin on each side
                if max_width > MIN_WINDOW_WIDTH:  # Ensure minimum window width
                    self.width = max_width
                    print(f"Reduced window width to {self.width:.2f}m to fit on wall")
                else:
                    self.is_valid = False
                    return False
            else:
                self.is_valid = False
                return False
            
        # Check if the cutout is too close to a corner
        # Project point onto wall line
        wall_vec = np.array([wall_end[0] - wall_start[0], wall_end[1] - wall_start[1]])
        wall_length = np.linalg.norm(wall_vec)
        wall_unit_vec = wall_vec / wall_length
        
        point_vec = np.array([self.location[0] - wall_start[0], self.location[1] - wall_start[1]])
        projection = np.dot(point_vec, wall_unit_vec)
        self.projection_on_wall = projection
        
        # Minimum distance from corner
        min_dist_from_corner = self.width / 2 + MIN_CUTOUT_CORNER_MARGIN
        
        if projection < min_dist_from_corner or projection > wall_length - min_dist_from_corner:
            print(f"Warning: {self.cutout_type} at {self.location} is too close to a corner")
            self.is_valid = False
            return False
        
        # Check for overlaps with existing cutouts
        if existing_cutouts:
            for cutout in existing_cutouts:
                if cutout.closest_wall_index == self.closest_wall_index:
                    # Cutouts are on the same wall, check for overlap
                    if self._overlaps_with(cutout):
                        print(f"Warning: {self.cutout_type} at {self.location} overlaps with existing {cutout.cutout_type}")
                        self.is_valid = False
                        return False
            
        self.is_valid = True
        return True
    
    def _overlaps_with(self, other: 'Cutout') -> bool:
        """
        Check if this cutout overlaps with another cutout on the same wall.
        
        Args:
            other: Another Cutout object
            
        Returns:
            bool: True if cutouts overlap, False otherwise
        """
        # Minimum separation distance between cutouts (in meters)
        min_separation = MIN_CUTOUT_SEPARATION
        
        # Check vertical overlap first (for windows at different heights)
        if self.cutout_type == "window" and other.cutout_type == "window":
            # If windows are at different heights, they might not overlap
            self_top = self.bottom_height + self.height
            other_top = other.bottom_height + other.height
            
            # No vertical overlap if one is completely above the other
            if self.bottom_height >= other_top or other.bottom_height >= self_top:
                # Still require horizontal separation for structural integrity
                min_separation = 0.05  # Minimal separation for windows at different heights
        
        # Reduce separation for window-door combinations to allow windows near doors
        if (self.cutout_type == "window" and other.cutout_type == "door") or \
           (self.cutout_type == "door" and other.cutout_type == "window"):
            min_separation = MIN_WINDOW_DOOR_SEPARATION
        
        # Check horizontal overlap along the wall
        self_start = self.projection_on_wall - (self.width / 2) - min_separation
        self_end = self.projection_on_wall + (self.width / 2) + min_separation
        
        other_start = other.projection_on_wall - (other.width / 2) - min_separation
        other_end = other.projection_on_wall + (other.width / 2) + min_separation
        
        # Check if ranges overlap
        return not (self_end < other_start or self_start > other_end)
    
    def get_available_space(self, existing_cutouts: List['Cutout']) -> float:
        """
        Calculate the maximum available space for this cutout on its wall,
        considering other cutouts.
        
        Args:
            existing_cutouts: List of already placed cutouts
            
        Returns:
            float: Maximum available width in meters
        """
        # Get cutouts on the same wall
        same_wall_cutouts = [c for c in existing_cutouts if c.closest_wall_index == self.closest_wall_index]
        
        if not same_wall_cutouts:
            # No other cutouts on this wall, use wall length minus corner margins
            return self.wall_length - WALL_CORNER_MARGIN * 2  # Margin on each side
        
        # Find the closest cutouts to the left and right
        left_cutout = None
        right_cutout = None
        min_left_dist = float('inf')
        min_right_dist = float('inf')
        
        for cutout in same_wall_cutouts:
            dist = cutout.projection_on_wall - self.projection_on_wall
            if dist < 0 and abs(dist) < min_left_dist:  # Cutout is to the left
                min_left_dist = abs(dist)
                left_cutout = cutout
            elif dist > 0 and dist < min_right_dist:  # Cutout is to the right
                min_right_dist = dist
                right_cutout = cutout
        
        # Calculate available space
        left_boundary = WALL_CORNER_MARGIN if not left_cutout else left_cutout.projection_on_wall + (left_cutout.width / 2) + 0.3
        right_boundary = self.wall_length - WALL_CORNER_MARGIN if not right_cutout else right_cutout.projection_on_wall - (right_cutout.width / 2) - 0.3
        
        # Ensure boundaries are valid
        left_boundary = max(WALL_CORNER_MARGIN, left_boundary)
        right_boundary = min(self.wall_length - WALL_CORNER_MARGIN, right_boundary)
        
        # Calculate maximum width
        max_width = right_boundary - left_boundary
        
        return max(0, max_width)
        
    def adjust_to_wall(self, room_polygon: Polygon, existing_cutouts: List['Cutout'] = []) -> bool:
        """
        Adjust the cutout location to be properly positioned on the closest wall
        and avoid overlaps with existing cutouts.
        
        Args:
            room_polygon: Shapely Polygon representing the room
            existing_cutouts: List of already placed cutouts to check for overlaps
            
        Returns:
            bool: True if adjustment was successful, False otherwise
        """
        # Get the wall segment
        wall_segments = list(zip(room_polygon.exterior.coords[:-1], room_polygon.exterior.coords[1:]))
        wall_start, wall_end = wall_segments[self.closest_wall_index]
        
        # Project the point onto the wall
        wall_vec = np.array([wall_end[0] - wall_start[0], wall_end[1] - wall_start[1]])
        wall_length = np.linalg.norm(wall_vec)
        self.wall_length = float(wall_length)
        wall_unit_vec = wall_vec / wall_length
        
        point_vec = np.array([self.location[0] - wall_start[0], self.location[1] - wall_start[1]])
        projection = np.dot(point_vec, wall_unit_vec)
        
        # Minimum distance from corner
        min_dist_from_corner = self.width / 2 + MIN_CUTOUT_CORNER_MARGIN
        
        # Adjust projection if too close to corners
        if projection < min_dist_from_corner:
            projection = min_dist_from_corner
            print(f"Adjusted {self.cutout_type} to be {min_dist_from_corner}m from wall start")
        elif projection > wall_length - min_dist_from_corner:
            projection = wall_length - min_dist_from_corner
            print(f"Adjusted {self.cutout_type} to be {min_dist_from_corner}m from wall end")
        
        # If there are existing cutouts, try to find a position that doesn't overlap
        if existing_cutouts:
            # Get cutouts on the same wall
            same_wall_cutouts = [c for c in existing_cutouts if c.closest_wall_index == self.closest_wall_index]
            
            if same_wall_cutouts:
                # Try to find a non-overlapping position
                if not self._try_find_position(wall_start, wall_unit_vec, wall_length,
                                            min_dist_from_corner, same_wall_cutouts):
                    print(f"Warning: Could not find non-overlapping position for {self.cutout_type}")
                    return False
                

            
        # Calculate new location on the wall
        new_x = wall_start[0] + projection * wall_unit_vec[0]
        new_y = wall_start[1] + projection * wall_unit_vec[1]
        
        # Update location
        self.location = (new_x, new_y)
        self.projection_on_wall = float(projection) 
        print(f"{self.cutout_type.capitalize()} adjusted to wall at {self.location}")
        
        # Re-validate
        return self.validate(room_polygon, existing_cutouts)

    def _try_find_position(self, wall_start: Tuple[float, ...], wall_unit_vec: np.ndarray,
                          wall_length: float, min_dist_from_corner: float,
                          same_wall_cutouts: List['Cutout']) -> bool:
        """
        Try to find a non-overlapping position by systematically searching along the wall.

        Args:
            wall_start: Wall start coordinates
            wall_unit_vec: Wall direction unit vector
            wall_length: Wall length
            min_dist_from_corner: Minimum distance from wall corners
            same_wall_cutouts: Other cutouts on the same wall

        Returns:
            bool: True if a valid position was found
        """
        original_projection = self.projection_on_wall
        step = 0.2  # Step size in meters
        max_steps = 20

        # Generate search pattern: start at original, then alternate left/right
        search_offsets = [0.0] + [offset for i in range(1, max_steps + 1) for offset in [i * step, -i * step]]

        for offset in search_offsets:
            projection = original_projection + offset

            # Check bounds
            if projection < min_dist_from_corner or projection > wall_length - min_dist_from_corner:
                continue

            # Calculate new location
            new_x = wall_start[0] + projection * wall_unit_vec[0]
            new_y = wall_start[1] + projection * wall_unit_vec[1]
            self.location = (new_x, new_y)
            self.projection_on_wall = float(projection)

            # Check for overlaps
            if not any(self._overlaps_with(cutout) for cutout in same_wall_cutouts):
                print(f"{self.cutout_type.capitalize()} adjusted to avoid overlap at {self.location}")
                return True

        return False

def apply_cutout_from_object(
    cutout: Cutout,
    current_wall_box: trimesh.Trimesh,
    mid_point: np.ndarray,
    wall_dir: np.ndarray,
    wall_length: float,
    wall_height: float,
    wall_thickness: float,
    trans_matrix: np.ndarray,
    rot_matrix: np.ndarray,
    room_polygon: Polygon
) -> trimesh.Trimesh:
    """
    Apply a cutout object to the given wall mesh.
    
    Args:
        cutout: Cutout object containing type, location and dimensions
        current_wall_box: Current wall mesh
        mid_point: Midpoint of the wall segment
        wall_dir: Direction vector of the wall
        wall_length: Length of the wall
        wall_height: Height of the wall
        wall_thickness: Thickness of the wall
        trans_matrix: Translation matrix for the wall
        rot_matrix: Rotation matrix for the wall
        room_polygon: Room polygon
        
    Returns:
        trimesh.Trimesh: New wall mesh with cutout applied
    """
    if not cutout.is_valid:
        print(f"Warning: Skipping invalid {cutout.cutout_type} at {cutout.location}")
        return current_wall_box

    depth: float = wall_thickness * 6.0
    # Convert cutout location to world space
    cutout_world: np.ndarray = np.array([
        cutout.location[0],
        0,
        room_polygon.bounds[3] - cutout.location[1]
    ])
    wall_to_cutout: np.ndarray = cutout_world - mid_point
    cutout_local_x: float = float(np.dot(wall_to_cutout, wall_dir))
    half_length: float = wall_length / 2
    cutout_local_x = max(-half_length + cutout.width/2, min(cutout_local_x, half_length - cutout.width/2))

    # Ensure window height doesn't exceed wall height
    cutout_height = cutout.height
    cutout_bottom = cutout.bottom_height
    
    # Check if window would extend beyond wall height
    if cutout_bottom + cutout_height > wall_height:
        # Adjust height to fit within wall
        original_height = cutout_height
        cutout_height = wall_height - cutout_bottom
        print(f"Adjusted {cutout.cutout_type} height from {original_height:.2f}m to {cutout_height:.2f}m to fit within wall height")
    
    cutout_box: trimesh.Trimesh = trimesh.creation.box(
        extents=[cutout.width, cutout_height, depth]
    )
    cutout_box.visual.face_colors = [200, 200, 200, 255]
    
    # Position cutout in wall's local space
    cutout_local_translation: np.ndarray = np.array([
        cutout_local_x,
        -wall_height/2 + cutout_bottom + cutout_height/2,
        -wall_thickness/2
    ])
    local_matrix: np.ndarray = trimesh.transformations.translation_matrix(cutout_local_translation)
    cutout_box.apply_transform(local_matrix)
    cutout_box.apply_transform(rot_matrix)
    cutout_box.apply_transform(trans_matrix)

    # Ensure meshes are watertight before boolean operation
    if not current_wall_box.is_watertight:
        print(f"Warning: Wall mesh for {cutout.cutout_type} is not watertight. Attempting to repair.")
        current_wall_box.fill_holes()
        if not current_wall_box.is_watertight:
            print(f"ERROR: Wall mesh for {cutout.cutout_type} could not be repaired. Skipping cutout.")
            return current_wall_box

    if not cutout_box.is_watertight:
        print(f"Warning: Cutout_box for {cutout.cutout_type} is not watertight. Attempting to repair.")
        cutout_box.fill_holes()
        if not cutout_box.is_watertight:
            print(f"ERROR: Cutout_box for {cutout.cutout_type} could not be repaired. Skipping cutout.")
            return current_wall_box

    new_wall = trimesh.boolean.difference(
        [current_wall_box, cutout_box],
        engine="manifold"
    )
    if new_wall is None:
        print(f"ERROR: All boolean engines failed for {cutout.cutout_type} cutout - returning original wall")
        return current_wall_box
    elif isinstance(new_wall, list):
        return trimesh.util.concatenate(new_wall)
    else:
        return new_wall

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

def validate_and_place_cutouts(room_polygon: Polygon,
                              door_location: Tuple[float, float],
                              window_locations: Optional[List[Tuple[float, float]]] = None,
                              door_width: float = DOOR_WIDTH,
                              door_height: float = DOOR_HEIGHT,
                              try_alternative_walls: bool = False) -> Tuple[Cutout, List[Cutout], List[Cutout]]:
    """
    Unified function for placing and validating door and window cutouts.

    This consolidates the duplicate logic from:
    - Scene._validate_window_location() in manager.py
    - _place_cutouts() in scene_3d.py

    Args:
        room_polygon: Room boundary polygon
        door_location: (x, y) position for door
        window_locations: List of (x, y) positions for windows
        door_width: Width of door
        door_height: Height of door
        try_alternative_walls: Whether to try placing failed windows on other walls

    Returns:
        Tuple of (door_cutout, valid_windows, all_cutouts)
    """
    from hsm_core.constants import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_BOTTOM_HEIGHT

    all_cutouts = []

    # Create and validate door
    door = Cutout(
        location=door_location,
        cutout_type="door",
        width=door_width,
        height=door_height
    )

    if not door.validate(room_polygon):
        door.adjust_to_wall(room_polygon)

    if door.is_valid:
        all_cutouts.append(door)

    # Create and validate windows
    windows = []

    if window_locations:
        for i, window_pos in enumerate(window_locations):
            window = Cutout(
                location=window_pos,
                cutout_type="window",
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT,
                bottom_height=WINDOW_BOTTOM_HEIGHT
            )

            # Validate and adjust window position
            if not window.validate(room_polygon, all_cutouts):
                if not window.adjust_to_wall(room_polygon, all_cutouts):
                    if try_alternative_walls:
                        # Try to place on alternative wall
                        window = _try_place_on_alternative_wall(window, room_polygon, all_cutouts)
                    else:
                        print(f"Warning: Could not find valid position for window at {window_pos}")
                        continue

            if window.is_valid:
                windows.append(window)
                all_cutouts.append(window)

    return door, windows, all_cutouts

def _try_place_on_alternative_wall(window: Cutout, room_polygon: Polygon,
                                  all_cutouts: List[Cutout]) -> Cutout:
    """Try to place a failed window on an alternative wall."""
    from hsm_core.constants import WINDOW_WIDTH
    import math

    # Get room vertices to try alternative walls
    room_coords = list(room_polygon.exterior.coords[:-1])

    for j in range(len(room_coords)):
        # Skip walls that already have both door and window
        wall_has_door = any(c.cutout_type == "door" and c.closest_wall_index == j for c in all_cutouts)
        wall_has_window = any(c.cutout_type == "window" and c.closest_wall_index == j for c in all_cutouts)

        if wall_has_door and wall_has_window:
            continue

        p1 = room_coords[j]
        p2 = room_coords[(j + 1) % len(room_coords)]
        wall_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        if wall_length < WINDOW_WIDTH + 0.2:
            continue

        # Try middle of this wall
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        alt_window = Cutout(
            location=(mid_x, mid_y),
            cutout_type="window",
            width=min(WINDOW_WIDTH, wall_length * 0.6)
        )

        if alt_window.validate(room_polygon, all_cutouts) or alt_window.adjust_to_wall(room_polygon, all_cutouts):
            print(f"Window moved to alternative wall at {alt_window.location}")
            return alt_window

    return window  # Return original if no alternative found

# DEPRECATED: Use validate_and_place_cutouts instead
def _place_cutouts(room_polygon: Polygon, door_location: Tuple[float, float],
                   window_locations: Optional[List[Tuple[float, float]]],
                   door_width: float, door_height: float) -> Tuple[Cutout, List[Cutout], List[Cutout]]:
    """DEPRECATED: Use validate_and_place_cutouts instead."""
    return validate_and_place_cutouts(room_polygon, door_location, window_locations,
                                    door_width, door_height, try_alternative_walls=False)

def _apply_cutouts_to_wall(wall_box: trimesh.Trimesh, wall_index: int,
                          door: Cutout, windows: List[Cutout],
                          mid_point: np.ndarray, wall_dir: np.ndarray,
                          wall_length: float, wall_height: float,
                          wall_thickness: float, trans_matrix: np.ndarray,
                          rot_matrix: np.ndarray, room_polygon: Polygon) -> trimesh.Trimesh:
    """Apply door and window cutouts to a wall mesh."""
    # Apply door cutout if on this wall
    if door.is_valid and door.closest_wall_index == wall_index:
        try:
            wall_box = apply_cutout_from_object(
                cutout=door,
                current_wall_box=wall_box,
                mid_point=mid_point,
                wall_dir=wall_dir,
                wall_length=wall_length,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                trans_matrix=trans_matrix,
                rot_matrix=rot_matrix,
                room_polygon=room_polygon
            )
        except Exception as e:
            print(f"Warning: Door cutout failed - {str(e)}")

    # Apply window cutouts if on this wall
    wall_windows = [w for w in windows if w.is_valid and w.closest_wall_index == wall_index]
    logger.debug(f"Wall {wall_index} has {len(wall_windows)} windows to apply")

    for j, window in enumerate(wall_windows):
        logger.debug(f"Applying window {j+1} to wall {wall_index}")
        try:
            wall_box = apply_cutout_from_object(
                cutout=window,
                current_wall_box=wall_box,
                mid_point=mid_point,
                wall_dir=wall_dir,
                wall_length=wall_length,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                trans_matrix=trans_matrix,
                rot_matrix=rot_matrix,
                room_polygon=room_polygon
            )
            logger.debug(f"Successfully applied window {j+1} to wall {wall_index}")
        except Exception as e:
            print(f"ERROR: Window cutout failed on wall {wall_index} - {str(e)}")
            import traceback
            traceback.print_exc()

    return wall_box

def _get_wall_id(motif: SceneMotif) -> Optional[str]:
    """Extract wall ID from motif if it exists."""
    if motif.object_type == ObjectType.WALL:
        if hasattr(motif, 'wall_alignment_id') and motif.wall_alignment_id:
            return motif.wall_alignment_id
    return None

def _get_parent_name_for_small_motif(motif: SceneMotif,
                                   scene_context: Optional[Dict[int, Tuple[SceneMotif, ObjectSpec]]]) -> Optional[str]:
    """Determine parent name for small object motifs."""
    if motif.object_type != ObjectType.SMALL:
        return None

    # Try to get parent from placement data first
    placement_data = getattr(motif, 'placement_data', None)
    if placement_data and 'parent_object' in placement_data:
        parent_name = placement_data['parent_object']
        logger.debug(f"Small motif {motif.id} assigned to parent {parent_name}")
        return parent_name

    # Fallback to object specs lookup
    if not motif.object_specs or not scene_context:
        return None

    for spec in motif.object_specs:
        if hasattr(spec, 'parent_object') and spec.parent_object:
            parent_id = spec.parent_object
            parent_info = scene_context.get(parent_id)
            if parent_info:
                _, parent_spec = parent_info
                parent_name = parent_spec.name
                logger.debug(f"Small motif {motif.id} assigned to parent ID {parent_id} (name: {parent_name})")
                return parent_name
            else:
                # Fallback to generic name if not found in context
                parent_name = f"parent_{parent_id}"
                logger.debug(f"Small motif {motif.id} assigned to parent ID {parent_id} (fallback name: {parent_name})")
                return parent_name

    return None

def _process_child_motifs(scene_objects: List[SceneObject], motif: SceneMotif,
                         scene_context: Optional[Dict[int, Tuple[SceneMotif, ObjectSpec]]],
                         top_level_motif_id: Optional[str]) -> List[SceneObject]:
    """Process child motifs recursively and return all objects."""
    all_objects = list(scene_objects)

    for obj in scene_objects:
        if obj.child_motifs:
            for child_motif in obj.child_motifs:
                child_scene_objects = create_scene_objects_from_motif(
                    child_motif,
                    parent_obj=obj,
                    scene_context=scene_context,
                    top_level_motif_id=top_level_motif_id
                )
                all_objects.extend(child_scene_objects)

    return all_objects

def _adjust_motif_rotation(motif_rotation: float, object_type: ObjectType) -> float:
    """Adjust motif rotation based on object type."""
    if object_type == ObjectType.LARGE or object_type == ObjectType.WALL:
        # Add 180° to make 0° face south (-Z) instead of north (+Z)
        # Only apply adjustment for specific rotations (0° or 180°)
        if motif_rotation == 0 or motif_rotation == 180:
            return motif_rotation + 180
    return motif_rotation

def _calculate_world_transform(local_transform: np.ndarray, motif_rotation_matrix: np.ndarray,
                              motif_position: Tuple[float, float, float]) -> np.ndarray:
    """Calculate world transform from local transform, motif rotation, and position."""
    # Combine transforms: world = motif rotation * local transform
    world_transform = np.dot(motif_rotation_matrix, local_transform)

    # Add translation from motif position
    position_offset = np.array([motif_position[0], motif_position[1], motif_position[2]]) \
                     if len(motif_position) == 3 \
                     else np.array([motif_position[0], 0, motif_position[1]])
    world_transform[0:3, 3] += position_offset

    return world_transform

def _calculate_rotation_angle(rotation_matrix: np.ndarray) -> float:
    """Calculate rotation angle in XZ plane relative to -Z axis (south-facing default)."""
    # Calculate unnormalized world forward direction (object space forward is +Z)
    unnormalized_front = rotation_matrix[:, 2]
    world_front = unnormalized_front / np.linalg.norm(unnormalized_front)

    # Calculate rotation angle in the XZ plane relative to -Z axis
    return (np.degrees(np.arctan2(world_front[0], world_front[2])) % 360)

def _transfer_transforms_to_scene_object(obj: 'ArrangementObject', scene_obj: SceneObject, verbose: bool) -> None:
    """Transfer HSSD and rotation optimization transforms from arrangement object to scene object."""
    # Transfer HSSD transforms
    if obj.has_hssd_alignment():
        hssd_transform_info = obj.get_hssd_alignment_transform()
        if hssd_transform_info:
            scene_obj.add_transform('hssd_alignment', hssd_transform_info)
            if verbose:
                logger.debug(f"Transferred HSSD alignment transform to {scene_obj.name}")

    # Transfer rotation optimisation data
    if hasattr(obj, 'transform_tracker') and obj.transform_tracker is not None and obj.mesh_path:
        try:
            mesh_id_tmp = obj.transform_tracker.get_mesh_id_from_path(obj.mesh_path)
            rot_opt_info = obj.transform_tracker.get_transform_by_type(mesh_id_tmp, "rotation_optimization")
            if rot_opt_info is not None:
                scene_obj.add_transform('rotation_optimization', rot_opt_info)
                if verbose:
                    logger.debug(f"Transferred rotation optimisation transform to {scene_obj.name}")
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to transfer rotation optimisation data for {scene_obj.name}: {e}")

def _determine_world_position(obj: SceneObject, scene_placer: 'SceneObjectPlacer') -> np.ndarray:
    """Determine the world position for an object considering various coordinate systems."""
    # Check if object comes from spatially optimized motif (already in world coordinates)
    if hasattr(obj, 'motif_id') and obj.motif_id:
        parent_motif = None
        # Search through all available motifs to find the parent
        for scene_motif in getattr(scene_placer, '_scene_motifs', []):
            if scene_motif.id == obj.motif_id:
                parent_motif = scene_motif
                break

        # If motif is spatially optimized, object position is already in world coordinates
        if parent_motif and getattr(parent_motif, 'is_spatially_optimized', False):
            world_position = np.array(obj.position)
            logger.debug(f"Using optimized world coordinates for {obj.name}")
            return world_position

    # Standard room to world conversion
    return room_to_world(obj.position, scene_placer.scene_offset)

def _apply_position_overrides(obj: SceneObject, world_position: np.ndarray) -> np.ndarray:
    """Apply position overrides and legacy hard overrides."""
    # Override with pre-optimized world coordinates when supplied by spatial optimizer
    if getattr(obj, "optimized_world_pos", None) is not None:
        world_position = np.array(obj.optimized_world_pos)  # type: ignore[arg-type]
        logger.debug(f"Using pre-optimised world coordinates for {obj.name}")
        return world_position

    # Legacy hard overrides (kept as fallback for specific cases)
    if obj.obj_type == ObjectType.LARGE and world_position[1] < 0:
        logger.warning(f"Fallback: Adjusting Y position for LARGE object {obj.name} from {world_position[1]:.2f} to 0.0")
        world_position[1] = 0.0

    return world_position

def _add_front_arrow(scene_placer: 'SceneObjectPlacer', obj: SceneObject, world_position: np.ndarray) -> None:
    """Add front arrow visualization for the object."""
    angle_rad = np.radians(obj.rotation)
    front_vector = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
    obj.front_vector = front_vector.tolist()

    arrow_mesh = create_front_arrow(front_vector, length=0.5, thickness=0.05)
    arrow_translation = trimesh.transformations.translation_matrix(world_position)
    arrow_mesh.apply_transform(arrow_translation)
    scene_placer.scene.add_geometry(arrow_mesh, geom_name=f"{obj.name}_front_arrow")

def _create_placed_object_info(obj: SceneObject, world_position: np.ndarray) -> Dict:
    """Create dictionary with placed object information."""
    rot_opt_mat = getattr(obj, '_preprocessing_data', {}).get('rotation_optimization_matrix')
    return {
        "name": obj.name,
        "dimensions": obj.dimensions,
        "position": world_position,
        "rotation": obj.rotation,
        "mesh_path": str(obj.mesh_path),
        "id": Path(obj.mesh_path).stem,
        "transform_matrix": rot_opt_mat if rot_opt_mat is not None else None,
    }

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

def room_to_world(pos: Tuple[float, float, float], scene_offset: List[float]) -> np.ndarray:
    """
    Transform a point from room coordinates to world coordinates using a unified transformation.
    
    Args:
        pos: (x, y, z) in room space.
             - In Room Space, the origin is at the room's bottom-left corner;
               x increases rightward and z increases forward.
        scene_offset: Offset computed from room bounds
    
    Returns:
        np.ndarray: (x, y, z) in World Space, where:
            - X is right,
            - Y is upward (with a small constant added so objects sit on the floor),
            - Z is computed by negating the room space z and adding the corresponding offset.
    """
    return np.array([
        float(pos[0]) + scene_offset[0],
        float(pos[1]) + scene_offset[1],
        float(pos[2]) + scene_offset[2]
    ])

class SceneObjectPlacer:
    """Class handling object placement in 3D scenes.
    
    Follows the coordinate system conventions:
        - Objects are centered in XZ plane
        - Y is normalized to bottom
        - Rotations are around Y axis anti-clockwise
        - World space follows room convention
    """
    
    def __init__(self, floor_height: float = FLOOR_HEIGHT, room_height: float = WALL_HEIGHT):
        self.scene: trimesh.Scene = trimesh.Scene()
        self.floor_height: float = floor_height
        self.room_height: float = room_height
        self.placed_objects: List[Dict] = []
        self.scene_offset: List[float] = [0, 0, 0]
        self.door_location: Tuple[float, float] = (0, 0)  # Default door location
        self.door_cutout: Optional[Cutout] = None  # Store door cutout object
        self.window_cutouts: Optional[List[Cutout]] = None  # Store window cutout objects
        self.scene_context: Optional[Dict[int, Tuple[SceneMotif, ObjectSpec]]] = None  # Context for ObjectSpec lookups

    def set_scene_context(self, scene_context: Dict[int, Tuple[SceneMotif, ObjectSpec]]) -> None:
        """Set the scene context for ObjectSpec lookups.

        Args:
            scene_context: Dictionary mapping object IDs to (SceneMotif, ObjectSpec) tuples
        """
        self.scene_context = scene_context

    def create_room_geom(self, room_polygon: Polygon, door_location: Tuple[float, float] = (0, 0), 
                         window_location: Optional[List[Tuple[float, float]]] = None) -> Dict[str, trimesh.Trimesh]:
        """Create and add floor and walls to scene.
        
        Args:
            room_polygon: Room boundary as a shapely Polygon
            door_location: (x, y) position of the door
            window_location: List of (x, y) positions for windows
        """
        room_geometry = {}
        # Normalize room polygon to have bottom-left at (0,0)
        bounds = room_polygon.bounds
        normalized_polygon = shapely.affinity.translate(
            room_polygon,
            xoff=-bounds[0],
            yoff=-bounds[1]
        )
        
        # Translate door location to match the normalized polygon
        normalized_door_location: Tuple[float, float] = (door_location[0] - bounds[0], door_location[1] - bounds[1])
        self.door_location = normalized_door_location
        
        # Normalize window locations if provided
        normalized_window_location = None
        if window_location:
            normalized_window_location = [(w[0] - bounds[0], w[1] - bounds[1]) for w in window_location]
        
        print("-"*50)
        print(f"Room polygon: {room_polygon.exterior.xy}")
        # print(f"Room polygon bounds: {room_polygon.bounds}")
        print(f"Normalized polygon bounds: {normalized_polygon.bounds}")
        # print(f"Scene offset: {self.scene_offset}")
        print(f"Door location: {self.door_location}")
        if normalized_window_location:
            print(f"Window locations ({len(normalized_window_location)}): {normalized_window_location}")
        else:
            print("No windows")
        print("-"*50)
        
        # Add floor and walls using normalized polygon
        floor_mesh = create_floor_mesh(normalized_polygon, self.scene_offset)
        room_geometry["floor"] = floor_mesh
        
        wall_meshes = create_wall_meshes(
            room_polygon=normalized_polygon,
            scene_offset=self.scene_offset,
            door_location=self.door_location,  # now normalized door location
            floor_height=self.floor_height,
            window_location=normalized_window_location,
            scene_placer=self  # Pass self reference to store cutout objects
        )
        
        # Add wall meshes to scene
        for name, mesh in wall_meshes:
            room_geometry[name] = mesh
        
        for name, mesh in room_geometry.items():
            self.scene.add_geometry(mesh, geom_name=name)
        
        return room_geometry
    
    def preprocess_object_mesh(self, obj: SceneObject, verbose: bool = True) -> Optional[trimesh.Trimesh]:
        """
        Preprocess object mesh by loading, applying HSSD transforms, and normalizing.
        
        Args:
            obj: SceneObject to preprocess
            verbose: Whether to print debug information
            
        Returns:
            Preprocessed trimesh object or None if preprocessing fails
        """
        return preprocess_object_mesh(obj, verbose)

    def place_object(self, obj: SceneObject, verbose: bool = False, preprocessed_mesh: Optional[trimesh.Trimesh] = None) -> bool:
        """Place a single object in the scene."""
        try:
            print(f"\n{'='*50}")
            print(f"Attempting to place: {obj.name} ({obj.obj_type.name})")

            if obj.child_motifs:
                for motif in obj.child_motifs:
                    print(f"  Child object: {motif.id}")
            print(f"Initial Position: {obj.position} Initial Rotation: {obj.rotation}°")
            print(f"Dimensions (W,H,D): {obj.dimensions}")

            # Use preprocessed mesh if provided, otherwise preprocess now
            if preprocessed_mesh is not None:
                loaded_obj = preprocessed_mesh.copy()
                # Get transform matrix from preprocessing data
                transform_matrix = getattr(obj, '_preprocessing_data', {}).get('transform_matrix', None)
            else:
                # Fallback to inline preprocessing
                loaded_obj = preprocess_object_mesh(obj)
                if loaded_obj is None:
                    return False
                transform_matrix = getattr(obj, '_preprocessing_data', {}).get('transform_matrix', None)
            
            # Apply Y-axis rotation (counterclockwise from south)
            rotation_angle = obj.rotation % 360
            y_rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=np.radians(rotation_angle),
                direction=[0, 1, 0],
                point=[0, 0, 0]
            )
            loaded_obj.apply_transform(y_rotation_matrix)

            # Determine world position considering various coordinate systems
            world_position = _determine_world_position(obj, self)

            print(f"World Position (X,Y,Z): ({world_position[0]:.2f}, {world_position[1]:.2f}, {world_position[2]:.2f}) Final Rotation: {rotation_angle}°")

            # Apply final transformation and add to scene
            translation_matrix = trimesh.transformations.translation_matrix(world_position)
            loaded_obj.apply_transform(translation_matrix)
            self.scene.add_geometry(loaded_obj, geom_name=obj.name)

            # Add front arrow visualization
            _add_front_arrow(self, obj, world_position)

            # Store placed object info
            placed_obj = _create_placed_object_info(obj, world_position)
            self.placed_objects.append(placed_obj)
            
            return True
            
        except Exception as e:
            print(f"\nERROR placing {obj.name}:")
            print(f"Exception: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def place_objects(self, objects: List[SceneObject], preprocessed_meshes: Optional[Dict[str, trimesh.Trimesh]] = None) -> trimesh.Scene:
        """Place multiple objects in the scene."""
        for obj in objects:
            preprocessed_mesh = preprocessed_meshes.get(obj.name) if preprocessed_meshes else None
            self.place_object(obj, preprocessed_mesh=preprocessed_mesh)
        print("-"*50)
        print(f"Placed {len(self.placed_objects)} objects in the scene")
        print("-"*50)
        return self.scene
    
def create_floor_mesh(room_polygon: Polygon, scene_offset: List[float]) -> trimesh.Trimesh:
    """
    Create a trimesh mesh from the room polygon to use as the floor.
    The floor should be in the XZ plane with Y being up.
    
    Args:
        room_polygon: Shapely Polygon representing room shape
        
    Returns:
        trimesh.Trimesh: Floor mesh
    """
    vertices = np.array(room_polygon.exterior.coords[:-1])
    
    # Convert from XY to XZ coordinates
    vertices_2d = vertices.flatten()
    
    # Use earcut for triangulation
    import earcut.earcut as ec
    triangles = ec.earcut(vertices_2d)
    faces = np.array(triangles).reshape(-1, 3)
    
    # Create vertices in XZ plane (Y up)
    vertices_3d = np.zeros((len(vertices), 3))
    vertices_3d[:, 0] = vertices[:, 0]   # X stays X
    vertices_3d[:, 2] = vertices[:, 1]   # Y becomes positive Z (removed negation)
    vertices_3d[:, 1] = 0                # Y (up) is zero for floor
    
    # Create the mesh and fix face winding
    mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces)
    mesh.fix_normals()  # This will ensure consistent face orientation
    
    # Flip the mesh if normals are pointing down
    if mesh.face_normals[0][1] < 0:  # Check if Y component of normal is negative
        mesh.faces = np.fliplr(mesh.faces)  # Reverse vertex order in faces
    
    # Translate the mesh to the scene offset
    translation_matrix = trimesh.transformations.translation_matrix(scene_offset)
    mesh.apply_transform(translation_matrix)
    
    return mesh

def create_scene_objects_from_motif(
    motif: SceneMotif,
    parent_obj: Optional[SceneObject] = None,
    scene_context: Optional[Dict[int, Tuple[SceneMotif, ObjectSpec]]] = None,
    top_level_motif_id: Optional[str] = None
) -> List[SceneObject]:
    """
    Create SceneObject instances from a SceneMotif.
    
    This function processes a motif's arrangement and recursively processes
    any child motifs (like small objects) to create a flat list of SceneObjects
    for the entire hierarchy under the given motif.

    Args:
        motif: The motif to process
        parent_obj: The parent SceneObject if this is a child motif
        scene_context: Global context for looking up object specifications
        top_level_motif_id: The ID of the top-level motif in a recursive call.
        
    Returns:
        A list of all SceneObjects created from this motif and its children.
    """
    # Return optimized objects directly if available
    if motif.is_spatially_optimized and hasattr(motif, '_scene_objects') and motif._scene_objects:
        return list(motif._scene_objects.values())

    # Extract wall ID and parent name
    wall_id = _get_wall_id(motif)
    parent_name = _get_parent_name_for_small_motif(motif, scene_context)
    current_top_level_motif_id = top_level_motif_id or motif.id
    
    # Preserve child motif links from original objects
    original_objects_by_name = {obj.name: obj for obj in motif.objects}

    # Process objects in the motif's arrangement to get their transforms
    scene_objects = process_arrangement_objects(
        arrangement=motif.arrangement,
        motif_rotation=motif.rotation,
        motif_position=motif.position,
        object_type=motif.object_type,
        wall_id=wall_id,
        parent_name=parent_name,
        motif_id=motif.id  # Use the actual motif ID, not the top-level one
    )
    
    # Process child motifs and collect all objects
    all_objects_in_motif_and_children = _process_child_motifs(
        scene_objects, motif, scene_context, current_top_level_motif_id
    )

    # Store the direct objects in the motif object itself
    motif.add_objects(scene_objects)
    
    if parent_obj and motif not in parent_obj.child_motifs:
        parent_obj.child_motifs.append(motif)
    

    return all_objects_in_motif_and_children

def create_scene_from_motifs(
    scene_motifs: List[SceneMotif],
    room_polygon: Optional[Polygon] = None,
    door_location: Tuple[float, float] = (0, 0),
    window_location: Optional[List[Tuple[float, float]]] = None,
    floor_height: float = FLOOR_HEIGHT,
    room_height: float = WALL_HEIGHT,
    enable_spatial_optimization: bool = True,
    scene_manager = None,
    output_dir: Optional[str] = None
) -> Tuple[trimesh.Scene, SceneObjectPlacer]:
    """
    Create a 3D scene from a list of motifs.
    
    This function processes a list of motifs to create a trimesh scene, including room geometry,
    objects, and their placements. It supports spatial optimization and scene export.
    
    Args:
        scene_motifs: List of motifs to include in the scene
        room_polygon: Shapely Polygon defining the room shape
        door_location: (x, y) coordinates for the door
        window_location: List of (x, y) coordinates for windows
        floor_height: Height of the floor from origin
        room_height: Height of the room from floor
        enable_spatial_optimization: Whether to enable spatial optimization
        scene_manager: The scene manager instance for accessing scene properties
        output_dir: Directory for saving output files
        
    Returns:
        trimesh.Scene: The generated 3D scene
    """
    
    # Initialize scene placer
    scene_placer = SceneObjectPlacer(room_height=room_height)

    # Store scene motifs in placer for optimized coordinate checking
    scene_placer._scene_motifs = scene_motifs

    # Create room geometry
    if room_polygon:
        scene_placer.create_room_geom(room_polygon, door_location, window_location)

    # Build scene context for parent lookups
    scene_context: Dict[int, Tuple[SceneMotif, ObjectSpec]] = {}
    # Create a comprehensive context from all motifs, including children
    all_motifs_for_context = []
    for m in scene_motifs:
        all_motifs_for_context.append(m)
        for obj in m.objects:
            if obj.child_motifs:
                all_motifs_for_context.extend(obj.child_motifs)

    for motif in all_motifs_for_context:
        if hasattr(motif, 'object_specs') and motif.object_specs:
            for obj_spec in motif.object_specs:
                if hasattr(obj_spec, 'id') and obj_spec.id is not None:
                    try:
                        scene_context[int(obj_spec.id)] = (motif, obj_spec)
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping object spec with non-integer ID {obj_spec.id} in motif {motif.id}")
    
    logger.info(f"Built scene context with {len(scene_context)} object specs for parent lookups")

    # Set scene context on placer for ObjectSpec lookups
    scene_placer.set_scene_context(scene_context)

    # Process all top-level motifs and collect SceneObjects
    all_scene_objects: List[SceneObject] = []
    
    for motif in scene_motifs:
        # We only process top-level motifs here. Child motifs (small objects)
        # are processed recursively within `create_scene_objects_from_motif`.
        scene_objects = create_scene_objects_from_motif(motif, scene_context=scene_context)
        all_scene_objects.extend(scene_objects)

    print(f"Processed {len(all_scene_objects)} scene objects from all motifs")
    
    # Place all objects in the scene
    preprocessed_meshes: Dict[str, trimesh.Trimesh] = {}
    for obj in all_scene_objects:
        preprocessed_mesh = preprocess_object_mesh(obj, verbose=False)
        if preprocessed_mesh:
            preprocessed_meshes[obj.name] = preprocessed_mesh

    # Create scene with processed objects
    scene: trimesh.Scene = scene_placer.place_objects(all_scene_objects, preprocessed_meshes)

    # Apply global scene transform to flip along Z-axis for Trimesh camera alignment
    flip_z = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    scene.apply_transform(flip_z)
    
    return scene, scene_placer

def create_wall_meshes(
    room_polygon: Polygon,
    scene_offset: List[float],
    door_location: Tuple[float, float],
    window_location: Optional[List[Tuple[float, float]]] = None,
    wall_height: float = WALL_HEIGHT,
    wall_thickness: float = WALL_THICKNESS,
    door_height: float = DOOR_HEIGHT,
    door_width: float = DOOR_WIDTH,
    floor_height: float = FLOOR_HEIGHT,
    scene_placer: Optional['SceneObjectPlacer'] = None
) -> List[tuple[str, trimesh.Trimesh]]:
    """
    Create wall meshes for the room with door and window cutouts.

    Args:
        room_polygon: Room boundary as a shapely Polygon
        scene_offset: Offset for scene coordinates
        door_location: (x, y) position of the door
        window_location: List of (x, y) positions of windows
        wall_height: Height of the walls
        wall_thickness: Thickness of the walls
        door_height: Height of the door
        door_width: Width of the door
        floor_height: Height of the floor
        scene_placer: Optional SceneObjectPlacer to store cutout objects
        
    Returns:
        List of (name, mesh) tuples for wall geometry
    """
    wall_meshes: List[tuple[str, trimesh.Trimesh]] = []
    wall_segments = list(zip(room_polygon.exterior.coords[:-1], room_polygon.exterior.coords[1:]))

    # Place door and window cutouts
    door, windows, all_cutouts = validate_and_place_cutouts(
        room_polygon, door_location, window_location, door_width, door_height,
        try_alternative_walls=False
    )

    # Store cutouts in scene_placer if provided
    if scene_placer:
        scene_placer.door_cutout = door if door.is_valid else None
        if windows:
            scene_placer.window_cutouts = windows

    print("-"*50)
    print(f"Valid cutouts: {len(all_cutouts)} ({len([c for c in all_cutouts if c.cutout_type == 'door'])} doors, {len(windows)} windows)")

    if window_location and len(windows) < len(window_location):
        print(f"Warning: Could only place {len(windows)} of {len(window_location)} requested windows")
    print("-"*50)
    
    # Process each wall segment
    for i, (start, end) in enumerate(wall_segments):
        start_arr: np.ndarray = np.array([start[0], 0, start[1]], dtype=float)
        end_arr: np.ndarray = np.array([end[0], 0, end[1]], dtype=float)
        wall_length: float = np.linalg.norm(end_arr - start_arr)
        wall_vector: np.ndarray = end_arr - start_arr
        wall_angle: float = np.arctan2(wall_vector[2], wall_vector[0])
        mid_point: np.ndarray = (start_arr + end_arr) / 2
        
        # Compute outward normal in XZ plane (right-hand rule)
        wall_dir: np.ndarray = wall_vector / np.linalg.norm(wall_vector)
        normal: np.ndarray = np.array([-wall_dir[2], 0, wall_dir[0]])  # Cross product with Y
        
        # Create wall box [length, height, thickness]
        wall_box: trimesh.Trimesh = trimesh.creation.box(
            extents=[wall_length, wall_height, wall_thickness]
        )
        wall_box.visual.face_colors = [200, 200, 200, 255]
        
        # Rotate wall to align with segment
        rot_matrix: np.ndarray = trimesh.transformations.rotation_matrix(
            angle=wall_angle,
            direction=WORLD_UP
        )
        wall_box.apply_transform(rot_matrix)
        
        # Position wall in world space
        translation: np.ndarray = (
            mid_point +                          # Base position
            scene_offset +                       # Scene offset
            np.array([0, wall_height / 2, 0]) +  # Lift to half height
            normal * (wall_thickness / 2)        # Offset by half thickness
        )
        trans_matrix: np.ndarray = trimesh.transformations.translation_matrix(translation)
        wall_box.apply_transform(trans_matrix)

        # Apply cutouts to wall
        wall_box = _apply_cutouts_to_wall(
            wall_box, i, door, windows, mid_point, wall_dir, wall_length,
            wall_height, wall_thickness, trans_matrix, rot_matrix, room_polygon
        )

        wall_meshes.append((f"wall_{i}", wall_box))

    return wall_meshes

def process_arrangement_objects(arrangement: Arrangement, motif_rotation: float = 0, 
                              motif_position: Tuple[float, float, float] = (0, 0, 0),
                              object_type: ObjectType = ObjectType.UNDEFINED,
                              wall_id: Optional[str] = None,
                              parent_name: Optional[str] = None,
                              motif_id: Optional[str] = None,
                              verbose: bool = False) -> list[SceneObject]:
    """
    Process objects in an arrangement to extract their transforms and information.
    
    Coordinate System:
        - Right-handed coordinate system
        - X: Right direction (positive right)
        - Y: Up direction (positive up)
        - Z: Forward direction (positive out of screen)
        - Rotation: Counterclockwise around Y-axis, 0° faces -Z (south)
    
    Coordinate convention clarification:
        - The source GLB meshes are modelled with their origin at the *bottom centre*.
        - After local → motif → world transformation we record the **centroid** of the
          mesh in `SceneObject.position`.
        - Down-stream systems (spatial optimiser, renderer, etc.) must therefore obtain
          the true bottom-Y via `position[1] - height/2`.
    
    Assumptions:
        - Objects are initially aligned with world south (facing -Z)
        - Each object is first transformed by its local transform (no_scale_matrix),
          then by the group (motif) rotation
    """
    scene_objects: list[SceneObject] = []

    # Adjust motif rotation based on object type
    adjusted_motif_rotation = _adjust_motif_rotation(motif_rotation, object_type)

    # Compute the rotation matrix for the group rotation (Y-axis)
    motif_rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(adjusted_motif_rotation), [0, 1, 0]  # Rotate around Y axis anti-clockwise
    )

    if verbose:
        print(f"Original Motif Rotation: {motif_rotation}°")
        print(f"Adjusted Motif Rotation: {adjusted_motif_rotation}°")
        print(f"Rotation matrix:\n{motif_rotation_matrix}")
        print(f"Motif Position: {motif_position}")
        print(f"Processing arrangement objects: {arrangement.description}")
    
    for obj in arrangement.objs:
        if verbose:
            print(f"\nProcessing object: {obj.label}")
        
        # Get object's local transform
        local_transform: np.ndarray = (
            obj.bounding_box.no_scale_matrix
            if obj.bounding_box.no_scale_matrix is not None
            else np.eye(4)
        )

        if verbose:
            print(f"Local transform:\n{local_transform}")

        # Calculate world transform
        world_transform = _calculate_world_transform(local_transform, motif_rotation_matrix, motif_position)

        if verbose:
            print(f"World transform:\n{world_transform}")

        # Calculate center position and rotation
        center: np.ndarray = world_transform[0:3, 3]
        rotation_matrix: np.ndarray = world_transform[0:3, 0:3]
        rotation_angle: float = _calculate_rotation_angle(rotation_matrix)

        if verbose:
            print(f"Computed center: {center}")
            print(f"Computed rotation angle (from -Z): {rotation_angle}°")
            print("-"*50)
        
        # Create SceneObject instance
        scene_obj: SceneObject = SceneObject(
            name=obj.label,
            position=(center[0], center[1], center[2]),
            dimensions=(obj.bounding_box.half_size * 2),  # Convert half_size to full dimensions.
            rotation=rotation_angle,
            mesh_path=obj.mesh_path,
            obj_type=object_type,
            parent_name=parent_name,
            wall_id=wall_id,
            motif_id=motif_id,
            id=obj.id  # Set the ID from the arrangement object
        )

        # Transfer transforms from arrangement object to scene object
        _transfer_transforms_to_scene_object(obj, scene_obj, verbose)

        # Add to scene objects list
        scene_objects.append(scene_obj)
        if verbose:
            print(f"Created scene object: {scene_obj.name} with type {scene_obj.obj_type}")
    
    return scene_objects

def create_front_arrow(front_vector: np.ndarray, length: float = 0.5, thickness: float = 0.05) -> trimesh.Trimesh:
    """
    Create an arrow mesh pointing in the given front_vector.
    
    Args:
        front_vector: 3D numpy array indicating the arrow's direction.
        length: Total length of the arrow.
        thickness: Overall thickness (diameter) of the arrow shaft.
        
    Returns:
        trimesh.Trimesh: Combined mesh for the arrow.
    """
    # Define parts: shaft (cylinder) and arrowhead (cone)
    shaft_length: float = length * ARROW_SHAFT_PROPORTION
    cone_length: float = length * ARROW_HEAD_PROPORTION
    shaft_radius: float = thickness * 0.5
    cone_radius: float = thickness

    # Create shaft: a cylinder originally centered at the origin.
    shaft: trimesh.Trimesh = trimesh.creation.cylinder(
        radius=shaft_radius,
        height=shaft_length,
        sections=32
    )
    # Translate shaft so that its base is at origin and it extends in +Z.
    shaft.apply_translation([0, 0, shaft_length / 2])
    
    # Create arrowhead as a cone.
    cone: trimesh.Trimesh = trimesh.creation.cone(
        radius=cone_radius,
        height=cone_length,
        sections=32
    )
    # Translate cone so its base touches the top of the shaft.
    cone.apply_translation([0, 0, shaft_length + (cone_length / 2)])
    
    # Combine shaft and cone.
    arrow: trimesh.Trimesh = trimesh.util.concatenate([shaft, cone])
    
    # The arrow is by default oriented along +Z.
    # Compute rotation matrix to align +Z to the provided front_vector.
    default_dir: np.ndarray = np.array([0, 0, 1])
    target_dir: np.ndarray = front_vector / np.linalg.norm(front_vector)
    axis: np.ndarray = np.cross(default_dir, target_dir)
    if np.linalg.norm(axis) < 1e-6:
        # Vectors are parallel or anti-parallel.
        if np.dot(default_dir, target_dir) < 0:
            rot: np.ndarray = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        else:
            rot = np.eye(4)
    else:
        rot_angle: float = np.arccos(np.clip(np.dot(default_dir, target_dir), -1, 1))
        rot: np.ndarray = trimesh.transformations.rotation_matrix(rot_angle, axis)
    arrow.apply_transform(rot)
    
    # Optionally, color the arrow (red, with full opacity)
    if hasattr(arrow, 'visual'):
        arrow.visual.face_colors = [255, 0, 0, 255]
    
    return arrow

if __name__ == "__main__":
    from hsm_core.scene.manager import Scene
    from hsm_core.scene.visualization import SceneVisualizer
    
    scene_path = Path("/local-scratch/localhome/hip4/project/t2s/hsm/scene/result_49_03027/0302-1042_prompt_16/scene_state.json")
    # Create scene
    scene = Scene.from_scene_state(scene_path)
    scene.window_location = [(1.5, 1.5), (4, 3), (3,4), (4,0),(0,3)]
    # Render scene
    SceneVisualizer(scene).render(non_blocking=False)
    # scene.create_scene()
    scene.export(Path("/local-scratch/localhome/hip4/project/t2s/hsm/scene/result_debug/room_scene.glb"))
    # scene.save_state(Path("/local-scratch/localhome/hip4/project/t2s/hsm/scene/result_debug/scene_state.json"))
    scene.save_stk_state(Path("/local-scratch/localhome/hip4/project/t2s/hsm/scene/result_debug"))

