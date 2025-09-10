"""
Motif Utilities Module

This module contains utility functions for working with SceneMotif objects.
"""

from __future__ import annotations
from typing import Optional, List

from hsm_core.scene.core.motif import SceneMotif
from hsm_core.scene.core.objects import SceneObject
from hsm_core.scene.core.objecttype import ObjectType


def get_all_motifs_recursive(motifs: list[SceneMotif]) -> list[SceneMotif]:
    """Recursively collect all motifs from a list of motifs and their children."""
    all_motifs = []
    for motif in motifs:
        all_motifs.append(motif)
        # Check for child motifs in objects
        for obj in motif.objects:
            if hasattr(obj, 'child_motifs') and obj.child_motifs:
                all_motifs.extend(get_all_motifs_recursive(obj.child_motifs))
    return all_motifs


def get_all_motifs(scene) -> list[SceneMotif]:
    """Get all motifs in the scene, including nested child motifs."""
    return get_all_motifs_recursive(scene._scene_motifs)


def get_all_objects(scene) -> list[SceneObject]:
    """Get all scene objects from all motifs, including nested ones."""
    all_objects = []
    seen_ids = set()
    for motif in get_all_motifs(scene):
        for obj in motif.objects:
            if obj.id not in seen_ids:
                all_objects.append(obj)
                seen_ids.add(obj.id)
    return all_objects


def get_motifs_by_types(scene, object_types: list[ObjectType] | ObjectType) -> Optional[list[SceneMotif]]:
    """Get motifs by their types."""
    if isinstance(object_types, ObjectType):
        return filter_motifs_by_type(scene.scene_motifs, object_types)
    else:
        return [motif for motif in scene.scene_motifs if motif.object_type in object_types]


def filter_motifs_by_type(motifs: list[SceneMotif], object_type: Optional[ObjectType] = None) -> list[SceneMotif]:
    """Filter motifs by object type if specified."""
    if object_type:
        return [m for m in motifs if m.object_type == object_type]
    return motifs
