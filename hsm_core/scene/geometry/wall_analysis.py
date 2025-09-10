"""
Wall Analysis

This module contains wall data extraction, blocking area analysis, and wall visualization utilities.
"""

import math
from matplotlib import pyplot as plt
import numpy as np
from shapely import LineString, Point
from shapely.geometry import Polygon
from hsm_core.constants import *
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.utils import get_logger

logger = get_logger('scene.geometry.wall_analysis')


def visualize_wall_blocked_areas(room_polygon, wall_data, output_path=None):
    """
    Visualize the walls of a room and highlight blocked areas on each wall.

    Args:
        room_polygon (Polygon): Shapely Polygon representing the room
        wall_data (list): List of dictionaries containing wall information:
            - id (str): Wall identifier
            - start (tuple): (x, y) coordinates of wall start
            - end (tuple): (x, y) coordinates of wall end
            - length (float): Length of the wall
            - angle (float): Angle of the wall in degrees
            - blocked_areas (list): List of dictionaries with keys:
                - start: position along wall where blocking starts
                - end: position along wall where blocking ends
                - height: height of the blocking object
                - object_id: identifier of the blocking object
            - merged_blocked_areas (list, optional): List of (start, end) tuples of merged blocked areas
        output_path (str): Path to save the output visualization

    Returns:
        plt.Figure: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot room outline
    x, y = room_polygon.exterior.xy
    ax.plot(x, y, color='black', linewidth=2, label='Room Outline')
    ax.fill(x, y, alpha=0.1, fc='lightblue')

    # Set up colors for different walls
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

    # Plot each wall and its blocked areas
    for i, wall in enumerate(wall_data):
        wall_id = wall["id"]
        start = wall["start"]
        end = wall["end"]
        blocked_areas = wall.get("blocked_areas", [])
        merged_blocked_areas = wall.get("merged_blocked_areas", [])
        wall_color = colors[i % len(colors)]

        # Draw the wall
        ax.plot([start[0], end[0]], [start[1], end[1]], color=wall_color, linewidth=3, label=f"Wall {wall_id}")

        # Add wall label at midpoint
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Calculate perpendicular offset for the label
        wall_angle_rad = math.radians(wall["angle"])
        perp_angle_rad = wall_angle_rad + math.pi/2
        offset = 0.2  # offset distance in meters
        label_x = mid_x + offset * math.cos(perp_angle_rad)
        label_y = mid_y + offset * math.sin(perp_angle_rad)

        ax.text(label_x, label_y, wall_id, fontsize=12, color=wall_color,
                fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor=wall_color, alpha=0.7))

        # Draw merged blocked areas (if available) or convert detailed areas to simple format
        blocks_to_draw = []
        if merged_blocked_areas:
            blocks_to_draw = [(start, end) for start, end in merged_blocked_areas]
        elif blocked_areas:
            if isinstance(blocked_areas[0], dict):
                blocks_to_draw = [(block["start"], block["end"]) for block in blocked_areas]
            else:
                blocks_to_draw = blocked_areas

        # Draw blocked areas
        for block_start, block_end in blocks_to_draw:
            # Calculate the actual points on the wall
            block_start_x = start[0] + (block_start / wall["length"]) * (end[0] - start[0])
            block_start_y = start[1] + (block_start / wall["length"]) * (end[1] - start[1])
            block_end_x = start[0] + (block_end / wall["length"]) * (end[0] - start[0])
            block_end_y = start[1] + (block_end / wall["length"]) * (end[1] - start[1])

            # Draw thicker red line for blocked area
            ax.plot([block_start_x, block_end_x], [block_start_y, block_end_y],
                   color='red', linewidth=6, alpha=0.6)

            # Add a small marker at each end of blocked area
            ax.plot(block_start_x, block_start_y, 'rx', markersize=8)
            ax.plot(block_end_x, block_end_y, 'rx', markersize=8)

    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Room Walls with Blocked Areas Highlighted')

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    # Add a note about blocked areas
    ax.text(0.02, 0.02, 'Red sections: Areas blocked by furniture',
            transform=ax.transAxes, fontsize=12, color='red',
            bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Wall visualization saved to {output_path}")

    return fig


def extract_wall_data(room_polygon: Polygon, scene_motifs: list,
                    detection_distance: float = FURNITURE_DETECTION_DISTANCE_DEFAULT,
                    door_location: tuple[float, float] = None,
                    window_locations: list[tuple[float, float]] = None,
                    include_wall_objects: bool = True,
                    existing_wall_data: list = None,
                    room_height: float = DEFAULT_ROOM_HEIGHT,
                    min_width_threshold: float = MIN_WIDTH_THRESHOLD_DEFAULT,
                    edge_gap_threshold: float = EDGE_GAP_THRESHOLD_DEFAULT) -> list[dict]:
    """
    Extract wall data including blocked areas from a room and its objects.

    Args:
        room_polygon (Polygon): Shapely Polygon representing the room
        scene_motifs (list): List of scene motifs/objects to check for wall blocking
        detection_distance (float): Maximum distance from wall to consider an object as blocking (default: FURNITURE_DETECTION_DISTANCE_DEFAULT)
        door_location (tuple, optional): (x, y) coordinates of the door
        window_locations (list, optional): List of (x, y) coordinates of windows
        include_wall_objects (bool): Whether to include wall objects in scene_motifs
        existing_wall_data (list, optional): Existing wall data to update
        room_height (float): Height of the room (default: DEFAULT_ROOM_HEIGHT)
        min_width_threshold (float): Min width for blocks/gaps (default: MIN_WIDTH_THRESHOLD_DEFAULT)
        edge_gap_threshold (float): Min edge/inter-object gap (default: EDGE_GAP_THRESHOLD_DEFAULT)

    Returns:
        wall_data: List of dictionaries containing wall information:
            - id (str): Wall identifier
            - start (tuple): (x, y) coordinates of wall start
            - end (tuple): (x, y) coordinates of wall end
            - length (float): Length of the wall
            - angle (float): Angle of the wall in degrees
            - blocked_areas (list): List of dictionaries with keys:
                - start: position along wall where blocking starts
                - end: position along wall where blocking ends
                - height: height of the blocking object
                - object_id: identifier of the blocking object
            - available_percent (float): Percentage of wall that is available for placement
            - processed_blocks (list): Processed list of blocks including gaps.
    """

    # Use existing wall data if provided, otherwise initialize new wall data
    if existing_wall_data:
        wall_data = existing_wall_data
        logger.info("Using existing wall data")
    else:
        # Extract wall segments from room polygon
        wall_segments = []
        vertices = list(room_polygon.exterior.coords)
        for i in range(len(vertices) - 1):
            wall_segments.append((vertices[i], vertices[i+1]))

        # Initialize wall data
        wall_data = []
        for wall_idx, (wall_start, wall_end) in enumerate(wall_segments):
            wall_vector = np.array([wall_end[0] - wall_start[0], wall_end[1] - wall_start[1]])
            wall_length = np.linalg.norm(wall_vector)
            wall_angle = math.degrees(math.atan2(wall_vector[1], wall_vector[0]))

            wall_data.append({
                "id": f"wall_{wall_idx}",
                "start": wall_start,
                "end": wall_end,
                "length": wall_length,
                "angle": wall_angle,
                "blocked_areas": [],
                "thickness": 0.01  # Add wall thickness (same as WALL_THICKNESS constant)
            })

    # Process door location if provided
    if door_location:
        from .grid_utils import calculate_door_angle

        door_angle = calculate_door_angle(door_location, room_polygon)

        # Find which wall the door is on
        door_point = Point(door_location)
        for wall_idx, wall in enumerate(wall_data):
            wall_line = LineString([wall["start"], wall["end"]])
            if door_point.distance(wall_line) < 0.1:  # Door should be very close to wall
                # Calculate door position along wall
                wall_vector = np.array([wall["end"][0] - wall["start"][0], wall["end"][1] - wall["start"][1]])
                wall_unit = wall_vector / wall["length"]
                wall_start_vec = np.array(wall["start"])
                door_vec = np.array(door_location) - wall_start_vec
                projection = np.dot(door_vec, wall_unit)

                # Add door as a blocked area
                wall_data[wall_idx]["blocked_areas"].append({
                    "start": max(0, projection - DOOR_WIDTH/2),
                    "end": min(wall["length"], projection + DOOR_WIDTH/2),
                    "height": DOOR_HEIGHT,
                    "object_id": ID_DOOR,
                    "is_door": True
                })

    # Process windows if provided
    logger.info(f"Window Processing - {len(window_locations) if window_locations else 0} windows")
    if window_locations:
        for window_idx, window_location in enumerate(window_locations):
            logger.debug(f"Window {window_idx}: location {window_location}")
            window_point = Point(window_location)
            window_placed = False
            for wall_idx, wall in enumerate(wall_data):
                wall_line = LineString([wall["start"], wall["end"]])
                distance = window_point.distance(wall_line)
                if distance < 0.1:
                    wall_vector = np.array([wall["end"][0] - wall["start"][0], wall["end"][1] - wall["start"][1]])
                    wall_unit = wall_vector / wall["length"]
                    wall_start_vec = np.array(wall["start"])
                    window_vec = np.array(window_location) - wall_start_vec
                    projection = np.dot(window_vec, wall_unit)

                    logger.debug(f"    Adding window to wall {wall_idx} at projection {projection:.2f}m")
                    wall_data[wall_idx]["blocked_areas"].append({
                        "start": max(0, projection - WINDOW_WIDTH/2),
                        "end": min(wall["length"], projection + WINDOW_WIDTH/2),
                        "height": WINDOW_HEIGHT,
                        "sill_height": WINDOW_SILL_HEIGHT,
                        "object_id": f"{ID_WINDOW_PREFIX}{window_idx}",
                        "is_window": True
                    })
                    window_placed = True
                    break
            if not window_placed:
                logger.warning(f"  Window {window_idx} at {window_location} could not be placed on any wall")
    else:
        logger.info("  No window_locations provided to extract_wall_data function")

    # Process scene motifs for wall blocking
    logger.debug(f"Wall occupancy detection ")
    for motif in scene_motifs:
        logger.debug(f"Checking motif: {motif.id}, Type: {motif.object_type}")
        x, y, z = motif.position
        width, height, depth = motif.extents
        rotation = motif.rotation

        is_wall_object = motif.object_type == ObjectType.WALL

        if not include_wall_objects and is_wall_object:
            logger.debug(f"  Skipping wall object: {motif.id}")
            continue

        if not is_wall_object and motif.object_type != ObjectType.LARGE:
            logger.debug(f"  Skipping small object: {motif.id}")
            continue

        # Find which wall this object is close to
        for wall_idx, wall in enumerate(wall_data):
            wall_start = wall["start"]
            wall_end = wall["end"]

            wall_line = LineString([wall_start, wall_end])
            dist = wall_line.distance(Point(x, z))
            logger.debug(f"  Wall {wall_idx}: distance = {dist:.2f}m")

            detection_threshold = WALL_OBJECT_DETECTION_THRESHOLD if is_wall_object else detection_distance

            if dist < detection_threshold:
                wall_vector = np.array([wall_end[0] - wall_start[0], wall_end[1] - wall_start[1]])
                wall_unit = wall_vector / wall["length"]

                # Special handling for wall objects (e.g., paintings)
                if is_wall_object:
                    wall_start_vec = np.array(wall_start)
                    object_vec = np.array([x, z]) - wall_start_vec
                    projection = np.dot(object_vec, wall_unit)

                    object_width_on_wall = width
                    proj_start = max(0, projection - object_width_on_wall/2)
                    proj_end = min(wall_length, projection + object_width_on_wall/2)

                    logger.debug(f"  Adding wall object to wall {wall_idx}: {proj_start:.2f}m - {proj_end:.2f}m at height {y:.2f}m")
                    wall_data[wall_idx]["blocked_areas"].append({
                        "start": proj_start,
                        "end": proj_end,
                        "height": height,
                        "mount_height": y,
                        "object_id": motif.id,
                        "is_wall_object": True
                    })

                    break
                else:
                    from .grid_utils import create_rotated_bbox
                    object_corners = create_rotated_bbox(x, z, width, depth, rotation)

                    projections = []
                    wall_start_vec = np.array(wall_start)
                    wall_length = wall["length"]
                    for corner in object_corners:
                        corner_vec = np.array(corner) - wall_start_vec
                        projection = np.dot(corner_vec, wall_unit)
                        if -0.5 <= projection <= wall_length + 0.5:
                            projections.append(projection)

                    if projections:
                        min_proj = max(0, min(projections))
                        max_proj = min(wall_length, max(projections))
                        logger.debug(f"  Adding blocked area on wall {wall_idx}: {min_proj:.2f}m - {max_proj:.2f}m")
                        wall_data[wall_idx]["blocked_areas"].append({
                            "start": min_proj,
                            "end": max_proj,
                            "height": height,
                            "object_id": motif.id
                        })
                    else:
                        logger.debug(f"  No valid projections on wall {wall_idx}")

    # Sort blocked areas for each wall by start position AND perform processing
    for wall in wall_data:
        wall["blocked_areas"] = sorted(wall["blocked_areas"], key=lambda x: x["start"])

        wall_length = wall["length"]
        total_blocked_length = 0

        processed_blocks = []
        sorted_blocks = wall["blocked_areas"]

        # Process all blocks to handle small widths
        for block in sorted_blocks:
            processed_block = block.copy()
            block_width = processed_block["end"] - processed_block["start"]

            if processed_block.get("is_door") or processed_block.get("is_window") or processed_block.get("is_wall_object"):
                processed_blocks.append(processed_block)
                continue

            if block_width < min_width_threshold:
                logger.debug(f"  Small width block detected ({block_width:.2f}m): {processed_block['object_id']}")
                processed_block["height"] = room_height

            processed_blocks.append(processed_block)

        # Add gap blocks for small spaces between wall edges and objects or between objects
        gap_blocks = []

        # Add gap blocks for small spaces
        if processed_blocks:
            # Check for gap at the start of the wall
            if processed_blocks[0]["start"] > 0 and processed_blocks[0]["start"] < edge_gap_threshold:
                logger.debug(f"  Small edge gap detected at wall start: {processed_blocks[0]['start']:.2f}m")
                gap_blocks.append({
                    "start": 0,
                    "end": processed_blocks[0]["start"],
                    "height": room_height,
                    "object_id": ID_UNUSABLE_GAP,
                    "is_gap": True
                })

            # Check for gaps between objects
            for j in range(len(processed_blocks) - 1):
                gap_size = processed_blocks[j+1]["start"] - processed_blocks[j]["end"]
                if 0 < gap_size < edge_gap_threshold:
                    logger.debug(f"  Small gap detected between objects: {gap_size:.2f}m")
                    gap_blocks.append({
                        "start": processed_blocks[j]["end"],
                        "end": processed_blocks[j+1]["start"],
                        "height": room_height,
                        "object_id": ID_UNUSABLE_GAP,
                        "is_gap": True
                    })

            # Check for gap at the end of the wall
            if processed_blocks[-1]["end"] < wall_length and (wall_length - processed_blocks[-1]["end"]) < edge_gap_threshold:
                logger.debug(f"  Small edge gap detected at wall end: {(wall_length - processed_blocks[-1]['end']):.2f}m")
                gap_blocks.append({
                    "start": processed_blocks[-1]["end"],
                    "end": wall_length,
                    "height": room_height,
                    "object_id": ID_UNUSABLE_GAP,
                    "is_gap": True
                })

        # Add gaps to processed blocks and sort again
        processed_blocks.extend(gap_blocks)
        processed_blocks = sorted(processed_blocks, key=lambda b: b["start"])

        # Calculate total blocked length by merging overlapping intervals
        if processed_blocks:
            intervals_to_merge = [[b["start"], b["end"]] for b in processed_blocks]
            merged_intervals = [intervals_to_merge[0]]
            for current_start, current_end in intervals_to_merge[1:]:
                last_merged_start, last_merged_end = merged_intervals[-1]

                if current_start <= last_merged_end:
                    merged_intervals[-1][1] = max(last_merged_end, current_end)
                else:
                    merged_intervals.append([current_start, current_end])

            total_blocked_length = sum(end - start for start, end in merged_intervals)
        else:
            total_blocked_length = 0.0

        # Calculate available wall percentage
        available_percent = 100 * (1 - total_blocked_length / wall_length) if wall_length > 0 else 0.0
        wall["available_percent"] = available_percent
        wall["processed_blocks"] = processed_blocks

        logger.debug(f"  Wall {wall['id']}: Merged Blocked Length = {total_blocked_length:.2f}m, Available space = {available_percent:.1f}%")

    return wall_data


def visualize_walls_as_surfaces(wall_data: list[dict], room_height: float = DEFAULT_ROOM_HEIGHT,
                              output_path: str = None, height_threshold: float = HEIGHT_THRESHOLD_DEFAULT,
                              specific_wall_ids: list[str] = None,
                              add_grid_markers: bool = False) -> plt.Figure:
    """
    Visualize each wall as a separate 2D surface with blocked areas highlighted.
    Relies on 'processed_blocks' and 'available_percent' being present in wall_data.

    Args:
        wall_data (list): List of dictionaries containing wall information, including:
            - id (str): Wall identifier
            - start (tuple): (x, y) coordinates of wall start
            - end (tuple): (x, y) coordinates of wall end
            - length (float): Length of the wall
            - angle (float): Angle of the wall in degrees
            - processed_blocks (list): REQUIRED. Processed list of dictionaries with keys:
                - start: position along wall where blocking starts
                - end: position along wall where blocking ends
                - height: height of the blocking object
                - object_id: identifier of the blocking object
                - is_door (bool, optional): Whether this blocked area is a door
                - is_window (bool, optional): Whether this blocked area is a window
                - is_wall_object (bool, optional): Whether this blocked area is a wall-mounted object
                - mount_height (float, optional): Height of wall object from floor
                - sill_height (float, optional): Height of window sill from floor
                - is_gap (bool, optional): Whether this is an unusable gap
            - available_percent (float): REQUIRED. Percentage of wall that is available.
        room_height (float): Height of the room in meters (default: DEFAULT_ROOM_HEIGHT)
        output_path (str, optional): Path to save the output visualization
        height_threshold (float): Objects taller than this threshold visualized as full height (default: HEIGHT_THRESHOLD_DEFAULT)
        specific_wall_ids (list[str], optional): IDs of the walls to visualize
        add_grid_markers (bool): Whether to add coordinate markers to the grid.

    Returns:
        plt.Figure: The matplotlib figure object, or None if no valid walls are found.
    """
    # Convert single wall to list format if needed
    if isinstance(wall_data, dict):
        wall_data = [wall_data]

    logger.info("Wall Surfaces Visualization")

    valid_wall_data = []
    if specific_wall_ids:
        logger.info(f"Filtering for specific walls: {specific_wall_ids}")
        for wall in wall_data:
            if (isinstance(wall, dict) and
                wall.get("id") in specific_wall_ids and
                "processed_blocks" in wall and
                "available_percent" in wall):
                valid_wall_data.append(wall)
            elif isinstance(wall, dict) and wall.get("id") in specific_wall_ids:
                logger.warning(f"Wall {wall.get('id')} is missing required data. Skipping.")
    else:
        logger.info("Visualizing all walls with required data.")
        for wall in wall_data:
            if (isinstance(wall, dict) and
                "processed_blocks" in wall and
                "available_percent" in wall):
                valid_wall_data.append(wall)
            elif isinstance(wall, dict):
                logger.warning(f"Wall {wall.get('id', 'Unknown ID')} is missing required data. Skipping.")
            else:
                logger.warning(f"Invalid wall data item skipped: {wall}")


    if not valid_wall_data:
        logger.error("No valid walls with processed data found to visualize!")
        return None

    wall_data = valid_wall_data # Use only the valid walls
    num_walls = len(wall_data)
    logger.info(f"Visualizing {num_walls} walls.")

    # Print summary of walls being visualized
    for i, wall in enumerate(wall_data):
        wall_id = wall.get("id", f"wall_{i}")
        logger.info(f"Wall {i} ({wall_id}): Length={wall.get('length', 0):.2f}m, Processed Blocks: {len(wall.get('processed_blocks', []))}, Available: {wall.get('available_percent', 0):.1f}%")


    # Determine the number of walls to visualize

    # Calculate the grid layout for subplots
    if specific_wall_ids:
        ncols = 1
        nrows = len(specific_wall_ids)
    elif num_walls == 1:
        ncols = 1
        nrows = 1
    elif num_walls <= 4:
        ncols = 2
        nrows = 2
    else:
        ncols = 3
        nrows = (num_walls + 2) // 3

    # Create figure with subplots
    figsize = (8, 4 * nrows) if num_walls > 1 else (15, 8)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Process each wall
    for i, wall in enumerate(wall_data):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]

        wall_id = wall["id"]
        wall_length = wall["length"]
        wall_start = wall["start"]
        wall_end = wall["end"]
        wall_angle = wall["angle"]
        processed_blocks = wall["processed_blocks"]
        available_percent = wall["available_percent"]

        # Draw the wall as a rectangle
        rect = plt.Rectangle((0, 0), wall_length, room_height,
                            facecolor='lightgray',
                            edgecolor='black',
                            alpha=0.5)
        ax.add_patch(rect)

        # Draw the blocked areas
        for block in processed_blocks:
            block_start = block.get("start", 0)
            block_end = block.get("end", 0)
            block_height = block.get("height", room_height)
            is_gap = block.get("is_gap", False)
            is_door = block.get("is_door", False)
            is_window = block.get("is_window", False)
            is_wall_object = block.get("is_wall_object", False)

            # For visualization, we always map from left (0) to right (wall_length)
            # This gives a consistent view when looking at the wall from inside the room
            adj_block_start = block_start
            adj_block_end = block_end

            # Handle wall angle reversal for consistent "inside looking out" view
            if abs(wall_angle - 180) < 1 or abs(wall_angle + 180) < 1:  # West wall
                adj_block_start = wall_length - block_end
                adj_block_end = wall_length - block_start

            # Check if height exists before comparison
            block_is_full_height = block_height >= height_threshold if block_height is not None else False

            # Determine block type and styling
            if is_gap:
                block_type = "gap"
            elif is_door:
                block_type = "door"
            elif is_window:
                block_type = "window"
            elif is_wall_object:
                block_type = "wall_object"
            elif block_is_full_height:
                block_type = "full_block"
            else:
                block_type = "partial_block"

            block_color = VIS_WALL_SURFACE_COLORS.get(block_type, 'gray')
            block_alpha = VIS_WALL_SURFACE_ALPHAS.get(block_type, 0.5)

            # Set vertical position (y-coordinate)
            block_start_y = 0
            if is_window:
                block_start_y = block.get("sill_height", WINDOW_SILL_HEIGHT)
            elif is_wall_object:
                block_start_y = block.get("mount_height", 1.5)

            # Create the rectangle patch for the block
            rect = plt.Rectangle((adj_block_start, block_start_y),
                                adj_block_end - adj_block_start,
                                block_height,
                                facecolor=block_color,
                                edgecolor='black',
                                alpha=block_alpha)
            ax.add_patch(rect)

            # Add object label
            object_id = block.get("object_id", "unknown")
            label_text = object_id # Default label is object ID

            if is_gap:
                label_text = f"{LABEL_UNUSABLE_GAP}\n{(block_end - block_start):.2f}m"
            elif is_door:
                label_text = f"{LABEL_DOOR}\n{block_height:.1f}m"
            elif is_window:
                sill_height = block.get("sill_height", WINDOW_SILL_HEIGHT)
                label_text = f"{LABEL_WINDOW}\n{block_height:.1f}m\nSill: {sill_height:.1f}m"
            elif is_wall_object:
                label_text = f"{LABEL_WALL_OBJECT}\n{block_height:.1f}m\n{object_id}"
            else: # Regular furniture block
                block_label = LABEL_BLOCKED if block_is_full_height else LABEL_PARTIAL
                label_text = f"{block_label}\n{block_height:.1f}m\n{object_id}"

            # Calculate label position
            label_y_pos = block_start_y + (block_height / 2 if block_height is not None else 0)

            ax.text((adj_block_start + adj_block_end) / 2, label_y_pos,
                    label_text,
                    ha='center', va='center',
                    color='black',
                    fontsize=VIS_SURFACE_TEXT_SIZE, fontweight='bold')

        # Add label for available space
        ax.text(0.5, 0.9, f"Available: {available_percent:.1f}%",
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                ha='center', fontsize=10)

        # Set plot limits and labels
        ax.set_xlim(-0.5, wall_length + 0.5)
        ax.set_ylim(-0.5, max(room_height + 0.5, 3.0))
        ax.set_xlabel("Wall position (meters)")
        ax.set_ylabel("Height (meters)")

        # Add wall start/end coordinates
        start_label = f"Start: ({wall_start[0]:.1f},{wall_start[1]:.1f})"
        end_label = f"End: ({wall_end[0]:.1f},{wall_end[1]:.1f})"

        # Add directional indicator text
        direction = f"Angle: {wall_angle:.1f}°"
        if -10 < wall_angle < 10:
            direction = "East Wall (View West) | ← North | South →"
        elif 170 < abs(wall_angle) < 190:
            direction = "West Wall (View East) | ← South | North →"
        elif 80 < wall_angle < 100:
            direction = "North Wall (View South) | ← West | East →"
        elif -100 < wall_angle < -80:
            direction = "South Wall (View North) | ← East | West →"


        ax.text(wall_length/2, -0.3, direction, fontsize=VIS_SURFACE_DIRECTION_TEXT_SIZE, ha='center')
        ax.text(0, -0.1, start_label, fontsize=VIS_SURFACE_COORD_TEXT_SIZE, ha='left')
        ax.text(wall_length, -0.1, end_label, fontsize=VIS_SURFACE_COORD_TEXT_SIZE, ha='right')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add grid markers with coordinates if enabled
        if add_grid_markers:
            TEXT_SIZE = VIS_SURFACE_TEXT_SIZE

            # Ensure markers are within the main rectangle (0 to wall_length, 0 to room_height)
            marker_x_min = 0
            marker_x_max = wall_length
            marker_y_min = 0
            marker_y_max = room_height

            # Calculate marker positions, starting from 0
            x_markers = np.arange(marker_x_min, marker_x_max + VIS_GRID_MARKER_INTERVAL_X, VIS_GRID_MARKER_INTERVAL_X)
            y_markers = np.arange(marker_y_min, marker_y_max + VIS_GRID_MARKER_INTERVAL_Y, VIS_GRID_MARKER_INTERVAL_Y)

            for x_val in x_markers:
                for y_val in y_markers:
                    # Check if the marker is within the main wall area
                    if marker_x_min <= x_val <= marker_x_max and marker_y_min <= y_val <= marker_y_max:
                        # Add coordinate text slightly below the marker
                        ax.text(x_val, y_val - 0.05 * room_height, f'({x_val:.1f},{y_val:.1f})',
                               color='gray',
                               fontsize=TEXT_SIZE,
                               ha='center',
                               va='top', # Place text below the marker point
                               bbox=dict(facecolor='red',
                                         edgecolor='none',
                                         alpha=0.5))
                        # Add 'x' marker at the grid intersection
                        ax.plot(x_val, y_val, 'x', color='gray', markersize=5, alpha=0.6)

        # Set title with wall info
        ax.set_title(f"{wall_id} (Length: {wall_length:.2f}m)")

    # Hide any unused subplots
    for i in range(num_walls, nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis('off')

    # Add a title and legend
    plt.suptitle(f"Wall Surfaces with Blocked Areas (Height Threshold: {height_threshold:.1f}m)", fontsize=16)

    # Create custom legend using constants
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=VIS_WALL_SURFACE_COLORS["full_block"], alpha=VIS_WALL_SURFACE_ALPHAS["full_block"], edgecolor='k'),
        plt.Rectangle((0, 0), 1, 1, facecolor=VIS_WALL_SURFACE_COLORS["partial_block"], alpha=VIS_WALL_SURFACE_ALPHAS["partial_block"], edgecolor='k'),
        plt.Rectangle((0, 0), 1, 1, facecolor=VIS_WALL_SURFACE_COLORS["gap"], alpha=VIS_WALL_SURFACE_ALPHAS["gap"], edgecolor='k'),
        plt.Rectangle((0, 0), 1, 1, facecolor=VIS_WALL_SURFACE_COLORS["door"], alpha=VIS_WALL_SURFACE_ALPHAS["door"], edgecolor='k'),
        plt.Rectangle((0, 0), 1, 1, facecolor=VIS_WALL_SURFACE_COLORS["window"], alpha=VIS_WALL_SURFACE_ALPHAS["window"], edgecolor='k'),
        plt.Rectangle((0, 0), 1, 1, facecolor=VIS_WALL_SURFACE_COLORS["wall_object"], alpha=VIS_WALL_SURFACE_ALPHAS["wall_object"], edgecolor='k')
    ]
    legend_labels = [
        f'Full block (H ≥ {height_threshold}m)',
        f'Partial block (H < {height_threshold}m)',
        f'Unusable gap (< {EDGE_GAP_THRESHOLD_DEFAULT}m)',
        'Door',
        'Window',
        'Wall Object'
    ]

    fig.legend(legend_handles, legend_labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for suptitle

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Wall surfaces visualization saved to {output_path}")

    return fig
