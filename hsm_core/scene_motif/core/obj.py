from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import trimesh
import numpy as np
from .bounding_box import BoundingBox

if TYPE_CHECKING:
    from .arrangement import Arrangement
    from hsm_core.retrieval.utils.transform_tracker import TransformTracker, TransformInfo

class Obj:
    def __init__(self, 
                 label: str, 
                 bounding_box: BoundingBox,
                 id: str | int | None = None,
                 mesh: Optional[np.ndarray] = None,
                 mesh_path: Optional[str] = None,
                 description: str = "",
                 transform_matrix: Optional[np.ndarray] = None,
                 matrix_order: str = "F",
                 transform_tracker: Optional['TransformTracker'] = None) -> None:
        """
        Initialize an object.

        Args:
            label: string, the label of the object
            bounding_box: BoundingBox, the bounding box of the object
            id: str | int | None, the ID of the object
            mesh: Optional[np.ndarray], the mesh of the object
            mesh_path: Optional[str], the path to the mesh file
            description: Optional[str], the description of the object
            transform_matrix: Optional[np.ndarray], the transform matrix of the object
            matrix_order: string, the order of the matrix for reshape
            transform_tracker: Optional[TransformTracker], tracker for mesh transformations
        
        Returns:
            None
        """

        self.label = label
        self.bounding_box = bounding_box
        self.id = id
        self.mesh = mesh
        self.mesh_path = mesh_path
        self.description = description
        self._referenced_arrangement: Optional[Arrangement] = None
        self.transform_tracker = transform_tracker
        
        # Handle transform matrix initialization properly
        if transform_matrix is not None:
            # If transform_matrix is provided, use it and reshape if needed
            if isinstance(transform_matrix, (list, tuple)):
                self.transform_matrix = np.array(transform_matrix).astype(float).reshape(4, 4, order=matrix_order)
            else:
                self.transform_matrix = np.array(transform_matrix).astype(float)
        else:
            # If no transform_matrix provided, use identity matrix
            self.transform_matrix = np.eye(4)
            
    def __str__(self) -> str:
        return (f"Obj(label='{self.label}', id='{self.id}', mesh_path='{self.mesh_path}'), transform_matrix={self.transform_matrix}")

    def calculate_bounding_box(self) -> BoundingBox:
        """
        Calculate the bounding box of the object.

        Returns:
            BoundingBox: the bounding box of the object
        """
        if not self.has_mesh:
            raise ValueError("Cannot calculate bounding box: object has no mesh.")

        # Calculate bounds
        bounds = self.mesh.bounds
        centroid = (bounds[0] + bounds[1]) / 2
        half_size = (bounds[1] - bounds[0]) / 2

        # Calculate principal axes using PCA
        points = self.mesh.vertices - centroid
        _, eigenvectors = np.linalg.eigh(np.cov(points.T))
        rotation_matrix = eigenvectors.T[::-1]  # Reverse order to get principal axes

        return BoundingBox(centroid, half_size, rotation_matrix)
            
    @property
    def has_mesh(self) -> bool:
        """
        Check if the object has a mesh.
        
        Returns:
            bool, whether the object has a mesh
        """
        return self.mesh is not None

    @classmethod
    def from_obj_file(cls, file_path: str, label: str) -> 'Obj':
        """
        Create an Obj instance from an OBJ file.

        Args:
            file_path: string, the path to the OBJ file
            label: string, the label of the object

        Returns:
            Obj: an instance of the Obj class
        """
        # Load the mesh from the OBJ file
        mesh = trimesh.load_mesh(file_path, force="mesh")

        # Calculate the bounding box
        bounds = mesh.bounds
        centroid = (bounds[0] + bounds[1]) / 2
        half_size = (bounds[1] - bounds[0]) / 2
        bounding_box = BoundingBox(centroid, half_size, np.eye(3))

        # Create and return the Obj instance
        return cls(label, bounding_box, None, mesh, None)

    def load_mesh(self, geom_name: str = None):
        """Load mesh and assign the original geometry name."""
        if self.mesh is not None:
            self.mesh = trimesh.load(self._mesh_path, force='mesh')
        if self._mesh_path is not None:
            if geom_name:
                self.mesh.metadata['name'] = geom_name
    
    def has_hssd_alignment(self) -> bool:
        """Check if this object has HSSD alignment transformation applied."""
        if not hasattr(self, 'transform_tracker') or self.transform_tracker is None or self.mesh_path is None:
            return False
        
        mesh_id = self.transform_tracker.get_mesh_id_from_path(self.mesh_path)
        return self.transform_tracker.has_hssd_alignment(mesh_id)

    def get_hssd_alignment_transform(self) -> TransformInfo | None:
        """Get the HSSD alignment transform for this object."""
        if not self.has_hssd_alignment():
            return None
        
        mesh_id = self.transform_tracker.get_mesh_id_from_path(self.mesh_path)
        return self.transform_tracker.get_transform_by_type(mesh_id, "hssd_alignment")