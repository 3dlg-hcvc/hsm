"""
Geometry Utilities Module

This module contains utility functions for geometric operations in the scene.
"""

from __future__ import annotations
from typing import Tuple

from hsm_core.scene.geometry.cutout import Cutout


def normalize_vertices(scene) -> tuple[list[list[float]], float, float]:
    """Normalize room vertices by translating to origin."""
    if hasattr(scene, '_cached_normalized_vertices'):
        return scene._cached_normalized_vertices
    from shapely.geometry import Polygon
    polygon = Polygon(scene.room_vertices)
    minx = polygon.bounds[0]
    miny = polygon.bounds[1]
    normalized_room_vertices = [[v[0] - minx, v[1] - miny] for v in scene.room_vertices]
    scene._cached_normalized_vertices = (normalized_room_vertices, minx, miny)
    return scene._cached_normalized_vertices


def get_normalized_cutouts(scene, minx: float, miny: float) -> tuple[object, object]:
    """Get normalized cutouts (door and windows) relative to origin."""
    if hasattr(scene, '_cached_normalized_cutouts'):
        return scene._cached_normalized_cutouts
    door_cutout = getattr(scene.scene_placer, 'door_cutout', None)
    window_cutouts = getattr(scene.scene_placer, 'window_cutouts', None)
    # Door
    if door_cutout:
        normalized_door_cutout = Cutout(
            location=(door_cutout.location[0] - minx, door_cutout.location[1] - miny),
            cutout_type=door_cutout.cutout_type,
            width=door_cutout.width,
            height=door_cutout.height
        )
        normalized_door_cutout.closest_wall_index = door_cutout.closest_wall_index
        normalized_door_cutout.projection_on_wall = door_cutout.projection_on_wall
        door_location = normalized_door_cutout
    else:
        normalized_door_location = (
            (scene.door_location[0] - minx, scene.door_location[1] - miny)
            if scene.door_location is not None else None
        )
        door_location = normalized_door_location
    # Windows
    if window_cutouts:
        normalized_window_cutouts = []
        for cutout in window_cutouts:
            normalized_cutout = Cutout(
                location=(cutout.location[0] - minx, cutout.location[1] - miny),
                cutout_type=cutout.cutout_type,
                width=cutout.width,
                height=cutout.height,
                bottom_height=cutout.bottom_height
            )
            normalized_cutout.closest_wall_index = cutout.closest_wall_index
            normalized_cutout.projection_on_wall = cutout.projection_on_wall
            normalized_window_cutouts.append(normalized_cutout)
        window_locations = normalized_window_cutouts
    elif hasattr(scene, 'window_location') and scene.window_location:
        normalized_window_locations = [(w[0] - minx, w[1] - miny) for w in scene.window_location]
        window_locations = normalized_window_locations
    else:
        window_locations = None
    scene._cached_normalized_cutouts = (door_location, window_locations)
    return scene._cached_normalized_cutouts
