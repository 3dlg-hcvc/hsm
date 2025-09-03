"""Door and window cutout handling for 3D scenes."""

import logging
from shapely.geometry import Polygon, LineString, Point
import trimesh
import numpy as np
from typing import List, Dict, Literal, Optional, Tuple
import shapely.affinity
import math

from hsm_core.constants import DOOR_WIDTH, DOOR_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_BOTTOM_HEIGHT

logger = logging.getLogger(__name__)

# Cutout-related constants
MIN_CUTOUT_CORNER_MARGIN: float = 0.1  # Minimum distance from corner for cutouts
MIN_CUTOUT_SEPARATION: float = 0.2  # Minimum separation between cutouts
MIN_WINDOW_DOOR_SEPARATION: float = 0.1  # Reduced separation between windows and doors

WALL_CORNER_MARGIN: float = 0.2  # Margin from wall corners
CUTOUT_WALL_DISTANCE_THRESHOLD: float = 0.1  # Maximum distance from wall for cutout to be valid


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
