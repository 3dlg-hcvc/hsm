import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MatplotlibPath


def point_in_hull(point, hull_points):
    """
    Check if a point lies within a 2D convex hull
    """
    hull = ConvexHull(hull_points)
    new_point = np.array([[point[0], point[1]]])
    hull_path = MatplotlibPath(hull_points[hull.vertices])
    return hull_path.contains_points(new_point)[0]

def project_point_to_hull(point, hull_points):
    """
    Project a point outside the hull to the nearest point on the hull boundary
    """
    hull = ConvexHull(hull_points)
    hull_path = MatplotlibPath(hull_points[hull.vertices])
    
    if hull_path.contains_points([[point[0], point[1]]])[0]:
        return point
    
    # Find the nearest point on the hull boundary
    min_dist = float('inf')
    nearest_point = None
    
    # Check each edge of the hull
    for i in range(len(hull.vertices)):
        p1 = hull_points[hull.vertices[i]]
        p2 = hull_points[hull.vertices[(i + 1) % len(hull.vertices)]]
        
        # Project point onto line segment
        edge = p2 - p1
        edge_length = np.linalg.norm(edge)
        edge_unit = edge / edge_length
        
        vec_to_point = np.array([point[0], point[1]]) - p1
        projection_length = np.dot(vec_to_point, edge_unit)
        
        if 0 <= projection_length <= edge_length:
            projection = p1 + edge_unit * projection_length
            dist = np.linalg.norm(np.array([point[0], point[1]]) - projection)
            if dist < min_dist:
                min_dist = dist
                nearest_point = projection
        
        # Check distance to vertices
        dist_to_p1 = np.linalg.norm(np.array([point[0], point[1]]) - p1)
        if dist_to_p1 < min_dist:
            min_dist = dist_to_p1
            nearest_point = p1
    
    return np.array([nearest_point[0], nearest_point[1]])

def get_object_corners(obj_pos, obj_dims, rotation_angle):
    """
    Get the corners of an object's footprint after rotation
    
    Args:
        obj_pos: [x, z] center position
        obj_dims: [width, height, depth] dimensions
        rotation_angle: rotation in degrees
    Returns:
        np.array: 4x2 array of corner coordinates
    """
    # Create corner points relative to center (counter-clockwise)
    w, d = obj_dims[0]/2, obj_dims[2]/2  # Use width and depth
    corners = np.array([
        [-w, -d],  # back left
        [w, -d],   # back right
        [w, d],    # front right
        [-w, d]    # front left
    ])
    
    # Create rotation matrix
    theta = np.radians(rotation_angle)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate corners and add object position
    rotated_corners = np.dot(corners, rot_matrix.T) + obj_pos
    return rotated_corners

def point_in_polygon(point, polygon_corners):
    """
    Check if a point lies inside a polygon using ray casting algorithm
    
    Args:
        point: [x,z] point to check
        polygon_corners: Nx2 array of polygon corner coordinates
    Returns:
        bool: True if point is inside polygon
    """
    x, z = point
    n = len(polygon_corners)
    inside = False
    
    j = n - 1
    for i in range(n):
        if ((polygon_corners[i][1] > z) != (polygon_corners[j][1] > z) and
            (x < (polygon_corners[j][0] - polygon_corners[i][0]) * 
             (z - polygon_corners[i][1]) / (polygon_corners[j][1] - polygon_corners[i][1]) + 
             polygon_corners[i][0])):
            inside = not inside
        j = i
    
    return inside

def check_vertical_clearance(obj_dims, layer_data, layer_key):
    """
    Check if object fits within layer height constraints using dimensions
    
    Args:
        obj_dims: [width, height, depth] dimensions
        layer_data: dictionary of layer information
        layer_key: current layer key (e.g. 'layer_0')
    Returns:
        bool: True if object fits within height constraints
    """
    if not layer_data or layer_key not in layer_data:
        return True  # No layer data available, assume it fits
        
    space_above = layer_data[layer_key]['space_above']
    return obj_dims[1] <= space_above  # Use height from dimensions

def project_to_boundary_with_rotation(obj_pos, obj_dims, rotation_angle, hull_points):
    """
    Project object to nearest valid position on boundary that fits the entire object
    
    Args:
        obj_pos: [x, z] center position
        obj_dims: [width, height, depth] dimensions
        rotation_angle: rotation in degrees
        hull_points: points defining the surface boundary
    Returns:
        np.array: [x, z] projected position that fits object
    """
    # Get initial corners
    corners = get_object_corners(obj_pos, obj_dims, rotation_angle)
    
    # Create hull
    hull = ConvexHull(hull_points)
    hull_vertices = hull_points[hull.vertices]
    
    # Find the nearest point on hull boundary for each corner
    projected_corners = np.zeros_like(corners)
    for i, corner in enumerate(corners):
        min_dist = float('inf')
        nearest_point = None
        
        # Check each edge of the hull
        for j in range(len(hull_vertices)):
            p1 = hull_vertices[j]
            p2 = hull_vertices[(j + 1) % len(hull_vertices)]
            
            # Project point onto line segment
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)
            edge_unit = edge / edge_length
            
            vec_to_point = corner - p1
            projection_length = np.dot(vec_to_point, edge_unit)
            
            if 0 <= projection_length <= edge_length:
                projection = p1 + edge_unit * projection_length
                dist = np.linalg.norm(corner - projection)
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = projection
        
        projected_corners[i] = nearest_point if nearest_point is not None else corner
    
    # Calculate new center position that would place the object fully inside
    new_center = projected_corners.mean(axis=0)
    return new_center

def calculate_hull_area(hull_points):
    """
    Calculate the area of a convex hull using the shoelace formula (also known as surveyor's formula).
    
    Args:
        hull_points: Numpy array of shape (N, 2) containing hull vertices
    
    Returns:
        float: Area of the hull
    """
    x = hull_points[:, 0]
    y = hull_points[:, 1]
    
    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def calculate_surface_occupancy(surface_objects, hull_points):
    """
    Calculate the percentage of surface area occupied by objects.
    
    Args:
        surface_objects: List of objects placed on the surface
        hull_points: Hull points defining the surface boundary
    
    Returns:
        float: Percentage of surface area occupied (0-100)
    """
    # Calculate total surface area using hull points
    hull_area = calculate_hull_area(hull_points)
    
    # Calculate total area occupied by objects
    occupied_area = 0
    for obj in surface_objects:
        # Use object dimensions to calculate its footprint
        width = obj["dimensions"][0]
        depth = obj["dimensions"][2]
        obj_area = width * depth
        occupied_area += obj_area
    
    # Calculate occupancy percentage
    occupancy_percentage = (occupied_area / hull_area) * 100
    
    return occupancy_percentage


def check_surface_boundary(obj_pos, obj_dims, rotation_angle, hull_points, layer_data=None, layer_key=None):
    """
    Check if object fits within surface boundary considering layer constraints
    
    Args:
        obj_pos: [x,z] center position
        obj_dims: [width, height, depth] dimensions
        rotation_angle: rotation in degrees
        hull_points: base hull boundary points
        layer_data: dictionary of layer information
        layer_key: current layer key
    Returns:
        bool: True if object fits within boundary
    """
    # Get layer-specific boundary if available
    boundary_points = hull_points
    if layer_data and layer_key in layer_data:
        boundary_points = layer_data[layer_key].get('boundary_points', hull_points)
        surface_type = layer_data[layer_key].get('surface_type', 'horizontal')
        
        if surface_type == 'vertical':
            # For vertical surfaces, check height constraints
            if obj_dims[1] > layer_data[layer_key].get('height', float('inf')):
                return False
    
    # Get object corners after rotation
    corners = get_object_corners(obj_pos, obj_dims, rotation_angle)
    
    # Check if all corners are inside boundary
    hull = ConvexHull(boundary_points)
    hull_path = MatplotlibPath(boundary_points[hull.vertices])
    return hull_path.contains_points(corners).all()

def check_layer_collisions(obj_pos, obj_dims, rotation_angle, placed_objects, current_layer):
    """
    Check collisions with objects in current and lower layers
    
    Args:
        obj_pos: [x,z] center position
        obj_dims: [width, height, depth] dimensions
        rotation_angle: rotation in degrees
        placed_objects: list of previously placed objects
        current_layer: current layer key
    Returns:
        bool: True if collision detected
    """
    current_layer_num = int(current_layer.split('_')[1])
    obj_corners = get_object_corners(obj_pos, obj_dims, rotation_angle)
    
    for placed_obj in placed_objects:
        placed_layer_num = int(placed_obj['layer_key'].split('_')[1])
        
        # Only check collisions with objects in same or lower layers
        if placed_layer_num <= current_layer_num:
            placed_corners = get_object_corners(
                placed_obj['position'],
                placed_obj['dimensions'],
                placed_obj.get('rotation', 0)
            )
            
            # Check for polygon intersection
            if polygons_intersect(obj_corners, placed_corners):
                return True
    
    return False

def find_valid_support_points(obj_pos, obj_dims, rotation_angle, layer_data, layer_key):
    """
    Find valid support points for object placement
    
    Args:
        obj_pos: [x,z] center position
        obj_dims: [width, height, depth] dimensions
        rotation_angle: rotation in degrees
        layer_data: dictionary of layer information
        layer_key: current layer key
    Returns:
        list: Valid support points or None if not required/available
    """
    if not layer_data or layer_key not in layer_data:
        return None
        
    support_points = layer_data[layer_key].get('support_points', [])
    if not support_points:
        return None
        
    # Get object footprint
    corners = get_object_corners(obj_pos, obj_dims, rotation_angle)
    
    # Find support points within object footprint
    valid_points = []
    for point in support_points:
        if point_in_polygon(point, corners):
            valid_points.append(point)
            
    return valid_points if valid_points else None

def polygons_intersect(corners1, corners2):
    """Check if two polygons intersect using Separating Axis Theorem"""
    def get_axes(corners):
        axes = []
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal / np.linalg.norm(normal))
        return axes
    
    def project(corners, axis):
        dots = [np.dot(corner, axis) for corner in corners]
        return min(dots), max(dots)
    
    # Get axes from both polygons
    axes = get_axes(corners1) + get_axes(corners2)
    
    # Check for separation along each axis
    for axis in axes:
        p1_min, p1_max = project(corners1, axis)
        p2_min, p2_max = project(corners2, axis)
        
        if p1_max < p2_min or p2_max < p1_min:
            return False
            
    return True

