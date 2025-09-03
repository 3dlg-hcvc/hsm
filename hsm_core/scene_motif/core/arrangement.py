from __future__ import annotations
from pathlib import Path
import trimesh
import numpy as np
import pickle
from copy import deepcopy

from .obj import Obj
from .bounding_box import BoundingBox

class Arrangement:
    def __init__(self, objs: list[Obj], description: str, function_call: str|None = None, glb_path: str|None = None) -> None:
        self.objs = objs
        self.description = description
        self.function_call = function_call
        self.glb_path = glb_path
        self._bounding_box: BoundingBox | None = None
        self._extents: np.ndarray | None = None
        self.visualization_path: str = ""
        self.program_str: str = ""
        self.validation_res: dict = {}
        # self._update_bounding_box()
    
    def __str__(self) -> str:
        return f"Arrangement(description='{self.description}', object_count={len(self.objs)})"
    
    def get_extents(self, recalculate: bool = True):
        if self._extents is None or recalculate:
            self._update_bounding_box()
        return self._extents
    
    def _update_bounding_box(self) -> None:
        """Calculate the axis-aligned bounding box that encompasses all objects in the arrangement."""
        # Get axis-aligned bounding box from scene
        scene = self.to_scene()
        
        if len(scene.geometry) == 0:
            return None
        
        bbox = scene.bounding_box
        self._bounding_box = bbox.bounds
        self._extents = bbox.extents
    
    def normalize(self) -> None:
        """
        Normalize the arrangement's objects so that one of the objects is at the origin and the rest are relative to it.
        """
        
        # The first object will be the one at the origin
        origin_obj = self.objs[0]

        # Move all the other objects relative to the origin object
        for obj in self.objs[1:]:
            obj.bounding_box.centroid -= origin_obj.bounding_box.centroid
            obj.bounding_box.centroid = np.round(obj.bounding_box.centroid, 5)
        
        # Move the origin object to the origin
        origin_obj.bounding_box.centroid = np.array([0, 0, 0])

    def center(self) -> None:
        """
        Center the entire arrangement by moving all objects so that the arrangement's
        center is at the origin (0,0,0) in the x-z plane only,
        The bottom of the arrangement will be at y=0.
        """
        self._update_bounding_box()
        
        if self._bounding_box is None:
            return
        
        arrangement_center_xz = (self._bounding_box[0] + self._bounding_box[1]) / 2
        arrangement_center_xz[1] = 0
        min_y = self._bounding_box[0][1]
        
        # Create the final offset vector
        # x and z components center the arrangement in the x-z plane
        # y component shifts the arrangement so its bottom is at y=0
        offset = np.array([arrangement_center_xz[0], min_y, arrangement_center_xz[2]])
        
        for obj in self.objs:
            obj.bounding_box.centroid -= offset
            obj.bounding_box.centroid = np.round(obj.bounding_box.centroid, 5)
            
        self._update_bounding_box()
    
    def to_mesh(self) -> trimesh.Trimesh:
        """
        Convert the arrangement to a trimesh.Trimesh object.

        Returns:
            trimesh.Trimesh: the trimesh object representing the arrangement
        """
        return trimesh.util.concatenate(self.to_scene().dump())
    
    def to_scene(self) -> trimesh.Scene:
        """
        Convert the arrangement to a trimesh.Scene object.

        Returns:
            trimesh.Scene: the trimesh scene object representing the arrangement
        """
        scene = trimesh.Scene()
        
        # Track used names to handle duplicates
        used_names = set()
        
        for obj in self.objs:           
            # Generate unique name for duplicate objects
            base_name = obj.label
            unique_name = base_name
            counter = 1
            while unique_name in used_names:
                unique_name = f"{base_name}_{counter}"
                counter += 1
            used_names.add(unique_name)
            
            # Update object label to unique name
            obj.label = unique_name
            
            # Create a copy of the mesh and apply transform
            if obj.mesh is not None:
                mesh = deepcopy(obj.mesh)
                transform = obj.bounding_box.no_scale_matrix
            else:
                raise Exception(f"Mesh is None for object {obj.label}")

            scene.add_geometry(
                mesh,
                geom_name=unique_name,
                node_name=unique_name,
                transform=transform
            )
            
        return scene
    
    def save(self, file_path: str = "saved_arrangement.glb") -> None:
        """
        Save the arrangement to a .glb file.

        Args:
            file_path: string, the path to the file to save the arrangement to
        """
        try:
            self.center()
            self.to_scene().export(file_path)
            self.glb_path = file_path
            print(f"Arrangement saved to {file_path}")
        except Exception as e:
            print(f"Error saving arrangement to {file_path}: {e}")
            raise

    def save_pickle(self, file_path: str = "saved_arrangement.pkl") -> None:
        """
        Save the arrangement to a pickle file.

        Args:
            file_path: string, the path to the file to save the arrangement to
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Arrangement saved to {file_path}")

    @staticmethod
    def load_pickle(file_path: str) -> Arrangement:
        """
        Load the arrangement from a pickle file.

        Args:
            file_path: string, the path to the pickle file containing the arrangement

        Returns:
            Arrangement: the arrangement object
        """
        # try:
        with open(Path(file_path), 'rb') as f:
            arrangement = pickle.load(f)
            arrangement._transform_matrix = None 
            arrangement._bounding_box = None 
        return arrangement

    def __getstate__(self):
        """Customize serialization to exclude mesh data."""
        state = self.__dict__.copy()
        
        # Store mesh paths and geometry names for each object
        for obj in state['objs']:
            if hasattr(obj, 'mesh') and obj.mesh is not None:
                # Store the mesh path and geometry name
                obj._mesh_path = obj.mesh_path
                obj._geom_name = obj.label
                obj.mesh = None
        
        return state

    def __setstate__(self, state):
        """Restore object state during deserialization."""
        self.__dict__.update(state)
        
        # Reload meshes with their original geometry names
        for obj in self.objs:
            if hasattr(obj, '_mesh_path') and obj._mesh_path is not None:
                # First load the mesh
                obj.load_mesh()
                # Only set metadata if mesh loading was successful
                if obj.mesh is not None:
                    obj.mesh.metadata['name'] = obj._geom_name
                # Clean up temporary attributes
                delattr(obj, '_mesh_path')
                delattr(obj, '_geom_name')
