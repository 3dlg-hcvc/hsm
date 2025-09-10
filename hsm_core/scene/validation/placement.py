"""
Placement Validation Module

This module handles validation of door and window placements in the scene.
"""

from __future__ import annotations
from typing import Optional, List, Tuple

from hsm_core.scene.geometry.cutout import Cutout
from hsm_core.constants import DOOR_WIDTH, DOOR_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_BOTTOM_HEIGHT
from hsm_core.utils import get_logger

logger = get_logger('scene.validation.placement')


def validate_door_location(scene, door_location: tuple[float, float]) -> tuple[float, float]:
    """
    Validate that the door location is at least 0.5m away from any corner
    to ensure a 1m wide door can fit properly.

    Args:
        scene: The scene object
        door_location: The door location to validate

    Returns:
        Valid door location or adjusted location if too close to corners
    """
    door = Cutout(
        location=door_location,
        cutout_type="door",
        width=DOOR_WIDTH,
        height=DOOR_HEIGHT,
    )

    # Initial validation attempt
    if door.validate(scene.room_polygon):
        return door.location

    # If initial validation fails, log warning and try to adjust to the closest wall
    logger.warning(f"Initial door location {door_location} is invalid. Attempting to adjust to the closest wall.")
    if door.adjust_to_wall(scene.room_polygon):  # Pass None for existing_cutouts as this is the first door
        # adjust_to_wall internally calls validate again. If it returns True, the door.location is updated and valid.
        logger.info(f"Door successfully adjusted to a valid position on the closest wall: {door.location}")
        return door.location
    else:
        # If adjust_to_wall also fails, then proceed with the original fallback (longest wall midpoint)
        logger.warning(f"Could not find valid door location for {door_location} even after trying to adjust to the closest wall. Falling back to longest wall midpoint.")
        # Fallback to center of longest wall
        longest_wall = None
        max_length = 0

        for i in range(len(scene.room_vertices)):
            p1 = scene.room_vertices[i]
            p2 = scene.room_vertices[(i + 1) % len(scene.room_vertices)]
            length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5

            if length > max_length and length >= DOOR_WIDTH + 0.2:
                max_length = length
                longest_wall = (p1, p2)

        if longest_wall:
            mid_x = (longest_wall[0][0] + longest_wall[1][0]) / 2
            mid_y = (longest_wall[0][1] + longest_wall[1][1]) / 2
            logger.info(f"Door moved to longest wall midpoint ({mid_x:.2f}, {mid_y:.2f})")
            return (mid_x, mid_y)

        # use original location (which is known to be invalid at this point, but it's the final fallback)
        logger.warning(f"All fallback methods failed. Using original invalid door location {door_location}.")
        return door_location


def validate_window_location(scene, window_location: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    """
    Validate that window locations are properly positioned on walls and don't overlap
    with each other or the door.

    Args:
        scene: The scene object
        window_location: List of (x, y) coordinates for windows

    Returns:
        List of validated window locations
    """
    if not window_location:
        return []

    # Setup door as existing cutout
    door = create_door_cutout(scene)
    all_cutouts = [door] if door.is_valid else []
    validated_windows = []

    # Process each window
    for window_pos in window_location:
        window = create_window_cutout(window_pos)
        if try_validate_window(scene, window, all_cutouts):
            validated_windows.append(window.location)
            all_cutouts.append(window)
        else:
            # Try alternative placement
            alt_window = find_alternative_window_placement(scene, all_cutouts)
            if alt_window:
                validated_windows.append(alt_window.location)
                all_cutouts.append(alt_window)

    return validated_windows


def create_door_cutout(scene) -> Cutout:
    """Create and validate door cutout."""
    door = Cutout(
        location=scene.door_location if hasattr(scene, 'door_location') else (0, 0),
        cutout_type="door",
    )
    door.validate(scene.room_polygon)
    return door


def create_window_cutout(location: tuple[float, float]) -> Cutout:
    """Create window cutout with standard dimensions."""
    return Cutout(
        location=location,
        cutout_type="window",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        bottom_height=WINDOW_BOTTOM_HEIGHT,
    )


def try_validate_window(scene, window: Cutout, all_cutouts: list) -> bool:
    """Try to validate window, with fallback to wall adjustment."""
    if window.validate(scene.room_polygon, all_cutouts):
        return True
    return window.adjust_to_wall(scene.room_polygon, all_cutouts)


def find_alternative_window_placement(scene, all_cutouts: list) -> Optional[Cutout]:
    """Find alternative placement for window on available walls."""
    for j in range(len(scene.room_vertices)):
        if wall_has_both_cutouts(scene, j, all_cutouts):
            continue

        wall_length = calculate_wall_length(scene, j)
        if wall_length < WINDOW_WIDTH + 0.2:
            continue

        mid_point = calculate_wall_midpoint(scene, j)
        alt_window = Cutout(
            location=mid_point,
            cutout_type="window",
            width=min(WINDOW_WIDTH, wall_length * 0.6)
        )

        if try_validate_window(scene, alt_window, all_cutouts):
            return alt_window

    return None


def wall_has_both_cutouts(scene, wall_index: int, all_cutouts: list) -> bool:
    """Check if wall already has both door and window."""
    has_door = any(c.cutout_type == "door" and c.closest_wall_index == wall_index for c in all_cutouts)
    has_window = any(c.cutout_type == "window" and c.closest_wall_index == wall_index for c in all_cutouts)
    return has_door and has_window


def calculate_wall_length(scene, wall_index: int) -> float:
    """Calculate length of wall segment."""
    p1 = scene.room_vertices[wall_index]
    p2 = scene.room_vertices[(wall_index + 1) % len(scene.room_vertices)]
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5


def calculate_wall_midpoint(scene, wall_index: int) -> tuple[float, float]:
    """Calculate midpoint of wall segment."""
    p1 = scene.room_vertices[wall_index]
    p2 = scene.room_vertices[(wall_index + 1) % len(scene.room_vertices)]
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
