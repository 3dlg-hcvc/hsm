"""
Grid Utilities

This module contains grid generation, spatial calculations, and door/window positioning utilities.
"""

import random
import numpy as np
import math
from shapely.geometry import Polygon, Point, LineString
from hsm_core.constants import *
from hsm_core.utils import get_logger

logger = get_logger('scene.geometry.grid_utils')


def _get_room_vertices(room_polygon):
    """Extract vertices from room polygon, excluding the last point (same as first)."""
    return list(room_polygon.exterior.coords)[:-1]


def create_grid(room_polygon, grid_size=DEFAULT_GRID_SIZE, door_location=None):
    """
    Create a grid of points within the room polygon.

    Args:
        room_polygon (Polygon): Shapely Polygon representing the room.
        grid_size (float): Size of each grid cell.
        door_location (tuple): (x, y) coordinates of the door location.

    Returns:
        tuple: (grid, door_location, bounds)
    """
    bounds = room_polygon.bounds
    x_min, y_min, x_max, y_max = bounds

    x_range = np.arange(x_min, x_max + grid_size, grid_size)
    y_range = np.arange(y_min, y_max + grid_size, grid_size)

    grid = []
    for x in x_range:
        for y in y_range:
            if room_polygon.contains(Point(x, y)):
                grid.append((x, y))

    grid = np.array(grid)

    if door_location is None:
        perimeter_points = [point for point in grid if Point(point).distance(room_polygon.boundary) < grid_size]

        if not perimeter_points:
            logger.warning("No perimeter points found. Using room centroid.")
            door_location = room_polygon.centroid.coords[0]
        else:
            corner_threshold = grid_size * 2
            corners = _get_room_vertices(room_polygon)
            valid_points = [
                point for point in perimeter_points
                if all(Point(point).distance(Point(corner)) > corner_threshold for corner in corners)
            ]

            if not valid_points:
                logger.warning("No valid door locations found. Using a random perimeter point.")
                door_location = random.choice(perimeter_points)
            else:
                door_location = random.choice(valid_points)

        logger.info(f"Door location set to: {door_location}")
    else:
        door_location = np.array(door_location)

    return grid, door_location, bounds


def calculate_door_angle(door_location, room_polygon):
    """
    Calculate the door angle based on its position on the room perimeter.
    The door angle will be perpendicular to the wall it's on, facing inward.

    Args:
        door_location (tuple): (x, y) coordinates of the door
        room_polygon (Polygon): Shapely Polygon representing the room

    Returns:
        float: Door angle in degrees (0-360)
    """
    vertices = _get_room_vertices(room_polygon)
    door_point = Point(door_location)

    min_distance = float('inf')
    closest_wall_points = None

    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        wall_line = LineString([p1, p2])

        distance = door_point.distance(wall_line)
        if distance < min_distance:
            min_distance = distance
            closest_wall_points = (p1, p2)

    if closest_wall_points:
        p1, p2 = closest_wall_points
        wall_angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        door_angle = (wall_angle + 90) % 360

        test_distance = 0.1  # Small distance to test inward direction
        test_x = door_location[0] + test_distance * math.cos(math.radians(door_angle))
        test_y = door_location[1] + test_distance * math.sin(math.radians(door_angle))
        test_point = Point(test_x, test_y)

        if not room_polygon.contains(test_point):
            door_angle = (door_angle + 180) % 360

        return door_angle

    return 0  # Default angle if no wall is found


def calculate_free_space(room_polygon, motifs):
    """
    Calculate the percentage of free space in the room.

    The free space is computed as the room area minus the area occupied by
    objects (motifs). Each motif is represented as a rectangle (using its
    x–z footprint and rotation) that is rotated by its given rotation (in degrees).

    Args:
        room_polygon (Polygon): Shapely Polygon representing the room.
        motifs (list): List of objects with attributes:
            - position (tuple): (x, y, z) with x and z on the floor.
            - extents (tuple): (width, height, depth)
            - rotation (float): Rotation angle in degrees.

    Returns:
        float: Free space percentage (0 to 100).
    """
    from shapely.affinity import rotate
    from shapely.ops import unary_union

    if not motifs:
        return 100.0

    occupied_polys = []
    for motif in motifs:
        try:
            x = motif.position[0]
            z = motif.position[2]
            width = motif.extents[0]
            depth = motif.extents[2]
            poly = Polygon([
                (x - width/2, z - depth/2),
                (x + width/2, z - depth/2),
                (x + width/2, z + depth/2),
                (x - width/2, z + depth/2)
            ])
            rotated_poly = rotate(poly, motif.rotation, origin=(x, z), use_radians=False)
            occupied_polys.append(rotated_poly)
        except Exception as e:
            logger.error(f"Error processing motif {motif.id}: {e}")

    if not occupied_polys:
        return 100.0

    try:
        union_poly = unary_union(occupied_polys)
        occupied_area = union_poly.intersection(room_polygon).area
        free_space_percent = ((room_polygon.area - occupied_area) / room_polygon.area) * 100
        return free_space_percent
    except Exception as e:
        logger.error(f"Error in calculating occupancy: {e}")
        return 0.0


def create_rotated_bbox(cx: float, cy: float, width: float, depth: float, angle_deg: float) -> list:
    """
    Create a rotated bounding box with arbitrary angle

    Args:
        cx (float): Center x-coordinate
        cy (float): Center y-coordinate
        width (float): Width of the box
        depth (float): Depth of the box
        angle_deg (float): Rotation angle in degrees

    Returns:
        list: List of corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    angle_rad = math.radians(angle_deg)
    hw, hd = width/2, depth/2

    corners = [
        (-hw, -hd),  # bottom-left
        (hw, -hd),   # bottom-right
        (hw, hd),    # top-right
        (-hw, hd)    # top-left
    ]

    rotated_bbox = []
    for x, y in corners:
        rotated_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        rotated_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        rotated_bbox.append((cx + rotated_x, cy + rotated_y))

    return rotated_bbox
