# Geometry Processing Constants
NORMAL_HORIZONTAL_THRESHOLD = 0.95  # Threshold for horizontal surfaces
MIN_AREA = 0.01  # Minimum area in square meters (100cm²) for initial filtering
AREA_THRESHOLD = 0.02  # Minimum area threshold for layer filtering
MIN_DIMENSION = 0.1  # Minimum dimension threshold (10cm) for geometry validation
VERTEX_MERGE_THRESHOLD = 1e-4  # Threshold for merging duplicate vertices in simplification
MERGE_DISTANCE = 0.03  # Distance threshold for merging touching geometries

# Layer Analysis Constants
HEIGHT_THRESHOLD_FACTOR = 1/6  # Factor to determine height threshold relative to total height
COVERAGE_RATIO_THRESHOLD = 0.7  # Threshold for determining if a layer is covered by higher layers
SIZE_THRESHOLD = 0.05  # Minimum size threshold for filtering thin geometries
MIN_AVAILABLE_SPACE_ABOVE_LAYER = 0.05  # Minimum space (in meters) required above a layer for it to be considered valid

# Paper terminology constants
CLEARANCE_THRESH = MIN_AVAILABLE_SPACE_ABOVE_LAYER  # Clearance threshold h_i (paper terminology)
TOP_HEIGHT = 0.5  # Default clearance for top surfaces without ceiling (paper terminology)
MERGE_THRESH = MERGE_DISTANCE  # Distance threshold for merging support regions (paper terminology)

# Segmentation Constants
VERTICAL_SEARCH_MARGIN = 0.05  # Search radius for vertical surfaces
EPSILON = 0.01  # Small buffer for numerical stability in segmentation
Z_MARGIN = 1.0  # Margin for verifying if vertical surface spans full Z-range
MIN_SEPARATION = 0.03  # Minimum separation between x positions when merging

# Visualization Constants
PADDING = 0.1  # Padding factor for visualization bounds
DEFAULT_COLOR_ALPHA = 0.3  # Default transparency value for 3D visualization
MAX_FACES_PER_BATCH = 2048  # Limit faces per batch for visualization

# Color palette for surfaces
SURFACE_COLORS = [
    ('red', '#FFB3B3'), ('blue', '#B3B3FF'), ('green', '#B3FFB3'),
    ('purple', '#E6B3E6'), ('orange', '#FFE6CC'), ('brown', '#E6CCCC'),
    ('pink', '#FFE6E6'), ('cyan', '#B3FFFF'), ('magenta', '#FFB3FF'),
    ('lime', '#CCFFCC'), ('teal', '#B3E6E6'), ('indigo', '#CCB3E6'),
    ('maroon', '#E6B3B3'), ('navy', '#B3B3E6'), ('olive', '#E6E6B3'),
    ('coral', '#FFD1C1'), ('gold', '#FFF2B3'), ('violet', '#F7D1F7'),
    ('turquoise', '#B3F7F7'), ('tan', '#F2E6D9'), ('salmon', '#FFD1C9'),
    ('plum', '#F2D1F2'), ('orchid', '#F2C9F2'), ('khaki', '#F7F2CC'),
    ('crimson', '#F2B3C1')
] 