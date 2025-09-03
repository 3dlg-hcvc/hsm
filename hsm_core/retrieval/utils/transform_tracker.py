from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path


@dataclass
class TransformInfo:
    """Information about a transformation applied to a mesh."""
    transform_type: str  # "hssd_alignment", "rotation_optimization", "normalization"
    transform_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    applied_order: int = 0  # Order in which this transform was applied


class TransformTracker:
    """Tracks transformations applied to meshes during retrieval."""
    
    def __init__(self) -> None:
        self._transforms: Dict[str, List[TransformInfo]] = {} # mesh_id -> list of TransformInfo
        self._next_order: int = 0
    
    def add_transform(
        self, 
        mesh_id: str, 
        transform_type: str, 
        transform_matrix: Optional[np.ndarray] = None,
        **metadata
    ) -> None:
        """Add a transformation record for a mesh.
        
        Args:
            mesh_id: Unique identifier for the mesh
            transform_type: Type of transformation applied
            transform_matrix: The transformation matrix (optional)
            **metadata: Additional metadata about the transformation
        """
        if mesh_id not in self._transforms:
            self._transforms[mesh_id] = []
        
        transform_info = TransformInfo(
            transform_type=transform_type,
            transform_matrix=transform_matrix.copy() if transform_matrix is not None else None,
            metadata=metadata,
            applied_order=self._next_order
        )
        
        self._transforms[mesh_id].append(transform_info)
        self._next_order += 1
    
    def get_transforms(self, mesh_id: str) -> List[TransformInfo]:
        """Get all transformations applied to a mesh."""
        return self._transforms.get(mesh_id, [])
    
    def has_transform_type(self, mesh_id: str, transform_type: str) -> bool:
        """Check if a mesh has a specific type of transformation."""
        transforms = self.get_transforms(mesh_id)
        return any(t.transform_type == transform_type for t in transforms)
    
    def has_hssd_alignment(self, mesh_id: str) -> bool:
        """Check if a mesh has HSSD alignment transformation."""
        return self.has_transform_type(mesh_id, "hssd_alignment")

    def get_mesh_id_from_path(self, mesh_path: str) -> str:
        """Convert a mesh path to a mesh ID.

        Args:
            mesh_path: Path to the mesh file

        Returns:
            str: Unique mesh ID derived from the path
        """
        from pathlib import Path
        path = Path(mesh_path)
        # Use filename without extension as mesh ID
        return path.stem

    def get_transform_by_type(self, mesh_id: str, transform_type: str) -> Optional[TransformInfo]:
        """Get the most recent transformation of a specific type for a mesh.

        Args:
            mesh_id: Unique identifier for the mesh
            transform_type: Type of transformation to retrieve

        Returns:
            TransformInfo or None: The most recent transform of the specified type, or None if not found
        """
        transforms = self.get_transforms(mesh_id)
        # Return the most recent transform of the specified type
        for transform in reversed(transforms):
            if transform.transform_type == transform_type:
                return transform
        return None 