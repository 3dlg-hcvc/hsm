from copy import deepcopy
import numpy as np
from ..core.bounding_box import BoundingBox
from ..core.obj import Obj
from ..core.arrangement import Arrangement

def create(label: str|Obj|Arrangement, 
           half_size: list[float]) -> Obj|Arrangement:
    '''
    Create an object with a bounding box, initialized with the canonical coordinate system.

    Args:
        label: string, the label of the object
        half_size: 1x3 vector, the half size of the bounding box
    
    Returns:
        obj: Obj, the object
    '''
    # deepcopy the label to avoid issues with the same object being used in multiple places
    if not isinstance(label, str):
        return deepcopy(label)
    else:
        bounding_box = BoundingBox([0, 0, 0], half_size, [1, 0, 0, 0, 1, 0, 0, 0, 1])
        transform_matrix = np.eye(4)
        obj = Obj(label, bounding_box, None, None, None, "", transform_matrix)
        return obj

def move(obj: Obj|Arrangement, x: float, y: float, z: float) -> None:
    '''
    Move an object's bounding box to a new position.

    Args:
        obj: Obj, the object
        x: float, the x coordinate of the new position
        y: float, the y coordinate of the new position
        z: float, the z coordinate of the new position
    
    Returns:
        None
    '''
    if not isinstance(obj, Obj):
        for obj in obj.objs:
            move(obj, x, y, z)
    else:
        translation = np.array([x, y, z]).astype(float)
        obj.bounding_box.centroid += translation

def rotate(obj: Obj|Arrangement, axis: str, angle: float) -> None:
    '''
    Rotate an object's bounding box's coordinate system around an axis in the box's coordinate system.

    Args:
        obj: Obj, the object
        axis: string, the axis of rotation ("x", "y", or "z")
        angle: float, the angle of rotation in degrees
    
    Returns:
        None
    '''

    # Get the axis of rotation
    if not isinstance(obj, Obj):
        for obj in obj.objs:
            rotate(obj, axis, angle)
    else:
        match axis:
            case "x":
                axis_of_rotation = obj.bounding_box.coord_axes[:, 0]
            case "y":
                axis_of_rotation = obj.bounding_box.coord_axes[:, 1]
            case "z":
                axis_of_rotation = obj.bounding_box.coord_axes[:, 2]
            case _:
                raise ValueError("Invalid axis")
        
        angle = np.radians(angle)
        kx, ky, kz = axis_of_rotation
        k_matrix = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        rotation_matrix = np.eye(3) + np.sin(angle) * k_matrix + (1 - np.cos(angle)) * k_matrix @ k_matrix
        
        obj.bounding_box.coord_axes = rotation_matrix @ obj.bounding_box.coord_axes
