"""
Constants for HSM.
"""

# Room and object dimensions
FLOOR_HEIGHT: float = 0.0
OBJ_Y_OFFSET: float = 0.1
WALL_HEIGHT: float = 2.5
WALL_THICKNESS: float = 0.01
DOOR_HEIGHT: float = 2.0
DOOR_WIDTH: float = 0.9

WINDOW_HEIGHT: float = 1.2
WINDOW_WIDTH: float = 1.6
WINDOW_BOTTOM_HEIGHT: float = 1.0  # Height from floor to bottom of window

# Grid and room constants
DEFAULT_GRID_SIZE = 0.5  # m
DEFAULT_ROOM_HEIGHT = WALL_HEIGHT
WINDOW_SILL_HEIGHT = WINDOW_BOTTOM_HEIGHT
WALL_OBJECT_DETECTION_THRESHOLD = 0.1
FURNITURE_DETECTION_DISTANCE_DEFAULT = 1.2
MIN_WIDTH_THRESHOLD_DEFAULT = 0.3
EDGE_GAP_THRESHOLD_DEFAULT = 0.3
HEIGHT_THRESHOLD_DEFAULT = 1.5

# Visualization constants
VIS_TEXT_SIZE = 16
VIS_GRID_MARKER_INTERVAL = 1
VIS_WALL_SURFACE_COLORS = {
    "gap": 'purple',
    "door": 'blue',
    "window": 'skyblue',
    "wall_object": 'green',
    "full_block": 'red',
    "partial_block": 'orange',
}
VIS_WALL_SURFACE_ALPHAS = {
    "gap": 0.6,
    "door": 0.7,
    "window": 0.7,
    "wall_object": 0.7,
    "full_block": 0.7,
    "partial_block": 0.5,
}
VIS_GRID_MARKER_INTERVAL_X = 1.0
VIS_GRID_MARKER_INTERVAL_Y = 1.0
VIS_SURFACE_TEXT_SIZE = 8
VIS_SURFACE_DIRECTION_TEXT_SIZE = 8
VIS_SURFACE_COORD_TEXT_SIZE = 7

# Object IDs and labels
ID_DOOR = "door"
ID_WINDOW_PREFIX = "window_"
ID_UNUSABLE_GAP = "unusable_gap"
LABEL_UNUSABLE_GAP = "UNUSABLE"
LABEL_DOOR = "DOOR"
LABEL_WINDOW = "WINDOW"
LABEL_WALL_OBJECT = "WALL OBJ"
LABEL_BLOCKED = "BLOCKED"
LABEL_PARTIAL = "PARTIAL"

# Mesh processing constants
MIN_WINDOW_WIDTH: float = 0.5  # Minimum width for windows 