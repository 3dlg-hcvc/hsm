"""
Mesh Path Utilities

This module provides utility functions for constructing and working with HSSD mesh paths.
"""

from pathlib import Path
from typing import Union


def construct_hssd_mesh_path(hssd_dir_path: Union[str, Path], mesh_id: str) -> Path:
    """
    Construct the full path to an HSSD mesh file given the base directory and mesh ID.

    The HSSD dataset organizes meshes in a hierarchical structure where each mesh
    is stored in a subdirectory based on the first character of its ID.

    Args:
        hssd_dir_path: Base path to the HSSD directory (e.g., /path/to/hssd-models)
        mesh_id: The mesh ID (e.g., "4f557c5ba812d2e72caa48e3ec46969a81039fc6")

    Returns:
        Path object pointing to the .glb file for the given mesh ID

    Example:
        >>> hssd_path = Path("/path/to/hssd-models")
        >>> mesh_id = "4f557c5ba812d2e72caa48e3ec46969a81039fc6"
        >>> path = construct_hssd_mesh_path(hssd_path, mesh_id)
        >>> str(path)
        '/path/to/hssd-models/objects/4/4f557c5ba812d2e72caa48e3ec46969a81039fc6.glb'
    """
    hssd_dir_path = Path(hssd_dir_path)

    if not mesh_id:
        raise ValueError("Mesh ID cannot be empty")

    if "part" in mesh_id:
        strip_part_id = mesh_id.split("_part_")[0]
        return hssd_dir_path / "objects" / "decomposed" / strip_part_id / f"{mesh_id}.glb"
    # Construct path: hssd_dir_path / "objects" / mesh_id[0] / f"{mesh_id}.glb"
    return hssd_dir_path / "objects" / mesh_id[0] / f"{mesh_id}.glb"
