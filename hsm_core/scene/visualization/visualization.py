import collections
import math
from pathlib import Path
import sys
from typing import Optional, Tuple, Any, List
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import platform
import matplotlib
import pyrender
import numpy as np
from shapely import Point
from shapely.geometry import LineString
import trimesh

from hsm_core.scene.geometry.grid_utils import create_grid, calculate_free_space
from hsm_core.scene.geometry.room_geometry import create_custom_room
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.constants import *
from hsm_core.scene.io.export import export_scene, save_scene_state_to_file
from hsm_core.utils import get_logger

logger = get_logger('scene.visualization')

sys.path.append(str(Path(__file__).parent.parent))

class SceneVisualizer:
    """Handles visualization of scene data."""
    
    def __init__(self, scene: Any):
        """
        Initialize SceneVisualizer with a scene object.

        Args:
            scene: Scene object to visualize
        """
        self.scene = scene
        self._setup_matplotlib_backend()
        
    def _setup_matplotlib_backend(self):
        """Configure matplotlib backend based on environment."""
        self.is_headless = (
            'DISPLAY' not in os.environ or 
            'CONDA_DEFAULT_ENV' in os.environ or 
            platform.system() == 'Linux' and not os.environ.get('DISPLAY')
        )
        
        if self.is_headless:
            matplotlib.use('Agg')  # Use non-interactive backend for headless environments
        else:
            matplotlib.use('TkAgg')  # Use TkAgg for interactive environments

    def display_plot(self, output_path: Path) -> None:
        """
        Display plot in a Tkinter window.
        
        Args:
            output_path: Path to the plot image
        """
        root = tk.Tk()
        root.title("Updated Plot")

        # Create a new figure for displaying the saved image
        fig, ax = plt.subplots(figsize=(10, 10))
        img = plt.imread(str(output_path))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        def on_closing():
            root.quit()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    def display_plot_async(self, output_path: Path) -> threading.Thread:
        """
        Display plot in a separate thread if not headless.
        
        Args:
            output_path: Path to the plot image
            
        Returns:
            threading.Thread: The display thread
        """
        if not self.is_headless:
            plot_thread = threading.Thread(
                target=self.display_plot,
                args=(output_path,),
                daemon=True
            )
            plot_thread.start()
            return plot_thread
        else:
            return None

    def save_plot(self, fig: plt.Figure, output_path: Path) -> None:
        """
        Save plot to file.
        
        Args:
            fig: matplotlib figure to save
            output_path: Path where to save the plot
        """
        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"Plot saved to {output_path}")

    def render(self, non_blocking: bool = True) -> Optional[threading.Thread]:
        """
        Render the scene using pyrenderer.
        
        Args:
            non_blocking (bool): If True, renders in a separate thread to avoid blocking
            
        Returns:
            Optional[threading.Thread]: Render thread if non_blocking is True
            
        Raises:
            RuntimeWarning: If rendering fails due to display connection issues
        """
        if self.scene.scene is None:
            self.scene.create_scene()
            
        if non_blocking:
            # Create a thread for rendering
            render_thread = threading.Thread(
                target=self._safe_render,
                args=(self.scene.scene, self.scene.room_vertices),
                daemon=True
            )
            render_thread.start()
            return render_thread
        else:
            # Blocking render
            self._safe_render(self.scene.scene, self.scene.room_vertices)

    def _safe_render(self, scene, vertices):
        """
        Internal method to safely render scene with error handling.
        
        Args:
            scene: Scene to render
            vertices: Room vertices
        """
        try:
            render_scene(scene, vertices)
        except Exception as e:
            logger.error(f"Render failed: {str(e)}")

    def visualize(self, output_path: Optional[str] = None, add_grid_markers: bool = False) -> Tuple[plt.Figure, float]:
        """Visualize the scene and optionally save to a file."""
        updated_plot, _, free_space_percent = visualize_grid(
            self.scene.room_polygon,
            grid_size=0.25,
            door_location=self.scene.door_location,
            motifs=self.scene.scene_motifs,
            add_grid_markers=add_grid_markers,
            window_location=self.scene.window_location
        )
        if output_path:
            # Save the figure to file
            updated_plot.savefig(output_path)
            # Create a copy of the figure before closing the original
            # This ensures we return a valid figure for later use
            fig_copy, ax_copy = plt.subplots(figsize=updated_plot.get_size_inches())
            # Copy the image data from the saved file
            import matplotlib.image as mpimg
            img = mpimg.imread(output_path)
            ax_copy.imshow(img)
            ax_copy.axis('off')
            plt.tight_layout()
            # Close the original figure to free memory
            plt.close(updated_plot)
            return fig_copy, free_space_percent
        return updated_plot, free_space_percent


def export_scene_glb(scene: Any, output_path: Path, recreate_scene: bool = False) -> None:
    """
    Export scene to GLB file.
    
    This is a wrapper function around Scene.export() to provide the expected
    interface for the pipeline and maintain backward compatibility.
    
    Args:
        scene: Scene object to export
        output_path: Path where to save the GLB file
        recreate_scene: Whether to recreate the scene before export
    """
    try:
        export_scene(scene, output_path, recreate_scene=recreate_scene)
        logger.info(f"Scene GLB exported to: {output_path}")
    except Exception as e:
        logger.error(f"Error exporting scene GLB: {e}")
        raise


def export_scene_state_to_json(scene: Any, output_path: Path) -> None:
    """
    Export scene state to JSON file.
    
    This is a wrapper function around Scene.save_state() to provide the expected
    interface for the pipeline and maintain backward compatibility.
    
    Args:
        scene: Scene object to export
        output_path: Path where to save the JSON state file
    """
    try:
        save_scene_state_to_file(scene, output_path)
        logger.info(f"Scene state exported to: {output_path}")
    except Exception as e:
        logger.error(f"Error exporting scene state: {e}")
        raise



def render_scene(trimesh_scene, room_vertices=None):
    # Create a pyrender scene
    scene = pyrender.Scene()
    
    if room_vertices is not None:
        room_vertices_np = np.array(room_vertices)
        center_x = np.mean(room_vertices_np[:, 0])
        center_y = np.mean(room_vertices_np[:, 1])
    else:
        center_x = 0
        center_y = 0

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[0, 3] = center_x  # Move the camera back along the x-axis
    camera_pose[1, 3] = center_y   # Move the camera right along the y-axis
    camera_pose[2, 3] = 8.0  # Adjust this value to move the camera up (along the z-axis in this system)
    scene.add(camera, pose=camera_pose)
    
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi/16.0,
                               outerConeAngle=np.pi/6.0)
    scene.add(light, pose=np.eye(4))

    for name, geometry in trimesh_scene.geometry.items():
        if isinstance(geometry, trimesh.PointCloud):
            # Handle point clouds
            points = geometry.vertices
            colors = geometry.colors if hasattr(geometry, 'colors') else None
            cloud_mesh = pyrender.Mesh.from_points(points, colors)
            scene.add(cloud_mesh)
        elif isinstance(geometry, trimesh.path.Path3D):
            # Convert Path3D to vertices and create a simple visualization
            vertices = geometry.vertices
            if vertices is not None and len(vertices) > 0:
                points = vertices
                path_mesh = pyrender.Mesh.from_points(points)
                scene.add(path_mesh)
        else:
            # Handle regular meshes
            mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
            scene.add(mesh)

    # Create a viewer with optimized settings
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False,
                             viewport_size=(800, 600))
    
def load_trimesh_scene(glb_path: str, verbose: bool = False) -> trimesh.Scene:
    if verbose:
        logger.info(f"Loading GLB scene from: {glb_path}")

    # Load GLB scene
    scene = trimesh.load(glb_path)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("File could not be loaded as a scene with multiple objects")

    if verbose:
        logger.info("Scene Graph Structure:")
        logger.info("----------------------")
        logger.info(f"Base frame: {scene.graph.base_frame}")
        logger.info(f"Number of nodes: {len(scene.graph.nodes)}")
        logger.info(f"Geometry nodes: {scene.graph.nodes_geometry}")
        logger.info("Geometry items:")
        for name, geom in scene.geometry.items():
            logger.info(f"- {name}: {type(geom)}")

    # Get all transforms and use their names as the original names
    transforms = {}
    
    # First, handle geometry that exists in the scene graph
    for node_name in scene.graph.nodes_geometry:
        if verbose:
            logger.info(f"Processing node: {node_name}")
        try:
            transform = scene.graph.get(node_name)[0]
            transforms[node_name] = transform
            if verbose:
                logger.info(f"Transform found for {node_name}")
        except Exception as e:
            if verbose:
                logger.error(f"Error getting transform for {node_name}: {e}")
            transforms[node_name] = np.eye(4)

    # Then, handle any remaining geometry that might be duplicates
    for geom_name in scene.geometry.keys():
        if geom_name not in transforms:
            if verbose:
                logger.info(f"Processing duplicate geometry: {geom_name}")
            original_name = geom_name.split('_')[0]
            if original_name in transforms:
                transforms[geom_name] = transforms[original_name].copy()
                if verbose:
                    logger.info(f"Using transform from {original_name} for {geom_name}")
            else:
                transforms[geom_name] = np.eye(4)
                if verbose:
                    logger.info(f"No original found for {geom_name}, using identity matrix")

    # Create new scene with transformed meshes
    transformed_scene = trimesh.Scene()
    all_vertices = []
    
    # Iterate directly over the geometry items
    for node_name, mesh in scene.geometry.items():
        mesh_transformed = mesh.copy()
        # Apply original transform from the scene graph
        mesh_transformed.apply_transform(transforms[node_name])
        
        # Collect all vertices for floor alignment
        all_vertices.extend(mesh_transformed.vertices)
        transformed_scene.add_geometry(mesh_transformed, geom_name=node_name)

    # Calculate the lowest point of all meshes
    all_vertices = np.array(all_vertices)
    min_y = np.min(all_vertices[:, 1]) if len(all_vertices) > 0 else 0
    
    if verbose:
        logger.info(f"Minimum y-coordinate: {min_y}")


    
    # Add an arrow to indicate the world origin
    origin_arrow = trimesh.creation.axis(origin_size=0.1)
    transformed_scene.add_geometry(origin_arrow, geom_name="world_origin")

    return transformed_scene, min_y


def export_scene_statistics(scene: Any, output_path: Path) -> None:
    """
    Export scene statistics to JSON file.
    
    Collects and exports comprehensive scene statistics including object counts,
    motif information, and spatial metrics.
    
    Args:
        scene: Scene object to analyze
        output_path: Path where to save the statistics JSON file
    """
    try:
        import json
        from datetime import datetime
        
        # Collect basic scene statistics
        all_motifs = scene.get_all_motifs() if hasattr(scene, 'get_all_motifs') else []
        all_objects = scene.get_all_objects() if hasattr(scene, 'get_all_objects') else []
        
        # Object type counts
        object_type_counts = {}
        if all_motifs:
            from collections import Counter
            type_counts = Counter(motif.object_type.name for motif in all_motifs if hasattr(motif, 'object_type'))
            object_type_counts = dict(type_counts)
        
        # Room geometry information
        room_info = {}
        if hasattr(scene, 'room_vertices') and scene.room_vertices:
            from shapely.geometry import Polygon
            room_polygon = Polygon(scene.room_vertices)
            room_info = {
                'area': float(room_polygon.area),
                'perimeter': float(room_polygon.length),
                'vertex_count': len(scene.room_vertices),
                'bounds': room_polygon.bounds
            }
        
        # Compile statistics
        statistics = {
            'timestamp': datetime.now().isoformat(),
            'scene_info': {
                'room_description': getattr(scene, 'room_description', ''),
                'room_type': getattr(scene, 'room_type', ''),
                'room_details': getattr(scene, 'room_details', '')
            },
            'counts': {
                'total_motifs': len(all_motifs),
                'total_objects': len(all_objects),
                'object_type_breakdown': object_type_counts
            },
            'room_geometry': room_info,
            'spatial_info': {
                'room_height': getattr(scene, 'room_height', 0),
                'door_location': getattr(scene, 'door_location', None),
                'window_location': getattr(scene, 'window_location', None)
            }
        }
        
        # Add motif-specific statistics
        if all_motifs:
            motif_stats = []
            for motif in all_motifs:
                motif_info = {
                    'id': getattr(motif, 'id', 'unknown'),
                    'object_type': getattr(motif, 'object_type', {}).name if hasattr(motif, 'object_type') else 'unknown',
                    'object_count': len(getattr(motif, 'objects', [])),
                    'has_position': hasattr(motif, 'position') and motif.position is not None,
                    'optimized': getattr(motif, 'optimized', False)
                }
                motif_stats.append(motif_info)
            statistics['motifs'] = motif_stats
        
        # Save statistics to file
        with open(output_path, 'w') as f:
            json.dump(statistics, f, indent=2, default=str)

        logger.info(f"Scene statistics exported to: {output_path}")

    except Exception as e:
        logger.error(f"Error exporting scene statistics: {e}")
        raise


def export_scene_debug_info(scene: Any, output_dir: Path) -> List[Path]:
    """
    Export comprehensive debug information for scene analysis.
    
    Creates multiple debug files with detailed scene information for
    development, debugging, and analysis purposes.
    
    Args:
        scene: Scene object to analyze
        output_dir: Directory where to save debug files
        
    Returns:
        List of paths to created debug files
    """
    try:
        import json
        from datetime import datetime
        
        debug_dir = output_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        created_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export detailed motif information
        if hasattr(scene, 'get_all_motifs'):
            motifs = scene.get_all_motifs()
            if motifs:
                motif_debug_path = debug_dir / f'motifs_debug_{timestamp}.json'
                motif_data = []
                
                for motif in motifs:
                    motif_info = {
                        'id': getattr(motif, 'id', 'unknown'),
                        'object_type': str(getattr(motif, 'object_type', 'unknown')),
                        'position': getattr(motif, 'position', None),
                        'optimized': getattr(motif, 'optimized', False),
                        'objects': []
                    }
                    
                    # Add object details
                    for obj in getattr(motif, 'objects', []):
                        obj_info = {
                            'id': getattr(obj, 'id', 'unknown'),
                            'position': getattr(obj, 'position', None),
                            'dimensions': getattr(obj, 'dimensions', None),
                            'has_mesh': hasattr(obj, 'mesh'),
                            'optimized_position': getattr(obj, 'optimized_world_pos', None)
                        }
                        motif_info['objects'].append(obj_info)
                    
                    motif_data.append(motif_info)
                
                with open(motif_debug_path, 'w') as f:
                    json.dump(motif_data, f, indent=2, default=str)
                created_files.append(motif_debug_path)
        
        # Export scene state dump
        scene_state_path = debug_dir / f'scene_state_dump_{timestamp}.json'
        scene_dump = {
            'timestamp': datetime.now().isoformat(),
            'scene_attributes': {},
            'room_geometry': {
                'vertices': getattr(scene, 'room_vertices', []),
                'height': getattr(scene, 'room_height', 0),
                'door_location': getattr(scene, 'door_location', None),
                'window_location': getattr(scene, 'window_location', None)
            },
            'scene_objects_count': len(scene.get_all_objects()) if hasattr(scene, 'get_all_objects') else 0,
            'has_3d_scene': hasattr(scene, 'scene') and scene.scene is not None,
            'has_scene_placer': hasattr(scene, 'scene_placer') and scene.scene_placer is not None
        }
        
        # Add basic scene attributes
        for attr in ['room_description', 'room_type', 'room_details', 'enable_spatial_optimization']:
            if hasattr(scene, attr):
                scene_dump['scene_attributes'][attr] = getattr(scene, attr)
        
        with open(scene_state_path, 'w') as f:
            json.dump(scene_dump, f, indent=2, default=str)
        created_files.append(scene_state_path)
        
        # Export spatial optimization info if available
        if hasattr(scene, 'scene_placer') and scene.scene_placer:
            spatial_debug_path = debug_dir / f'spatial_debug_{timestamp}.json'
            spatial_info = {
                'placed_objects_count': len(getattr(scene.scene_placer, 'placed_objects', [])),
                'optimizer_stats': getattr(scene.scene_placer, 'optimizer_stats', {}),
                'collision_detection_enabled': True  # Assume enabled if scene_placer exists
            }
            
            with open(spatial_debug_path, 'w') as f:
                json.dump(spatial_info, f, indent=2, default=str)
            created_files.append(spatial_debug_path)
        
        logger.info(f"Debug information exported to {len(created_files)} files in: {debug_dir}")
        return created_files

    except Exception as e:
        logger.error(f"Error exporting scene debug info: {e}")
        raise


def visualize_grid(room_polygon, grid_size: float = DEFAULT_GRID_SIZE,
                   door_location: tuple[float, float] = None,
                   window_location: list[tuple[float, float]] = None, motifs: list = None,
                   use_scene_objects: bool = False,
                   add_grid_markers: bool = True):
    """
    Visualize the room layout with grid and placed objects.

    Args:
        room_polygon (Polygon): Room shape
        grid_size (float): Size of grid cells (default: DEFAULT_GRID_SIZE)
        door_location (tuple): Door position
        window_location (list[tuple]): Window position
        motifs (list[SceneMotif], optional): List of SceneMotif objects. This is the primary source of
                                           objects for visualization.
        use_scene_objects (bool): If True, the function will attempt to visualize SceneObject instances
                                  contained within each motif (i.e., motif.scene_objects).
                                  If False (default), it visualizes the motifs themselves.
        add_grid_markers (bool): Whether to add coordinate markers

    Returns:
        plt.Figure: The matplotlib figure object
        door_location (tuple): Door position
        free_space_percent (float): Percentage of free space
    """
    TEXT_SIZE = VIS_TEXT_SIZE
    GRID_MARKER_INTERVAL = VIS_GRID_MARKER_INTERVAL
    LABEL_OFFSET = 0.2 # Offset for wall labels outside the room

    PlotItem = collections.namedtuple('PlotItem', ['id', 'position', 'extents', 'rotation', 'object_type'])
    items_to_process = []

    if use_scene_objects:
        if motifs:
            for m in motifs:
                if hasattr(m, 'scene_objects') and m.scene_objects:
                    for so in m.scene_objects:
                        # Ensure scene objects have the necessary attributes for PlotItem
                        if all(hasattr(so, attr) for attr in ['name', 'position', 'dimensions', 'rotation', 'object_type']):
                            items_to_process.append(PlotItem(id=so.name, position=so.position, extents=so.dimensions, rotation=so.rotation, object_type=so.object_type))
                        else:
                            logger.warning(f"SceneObject in motif {m.id} is missing attributes. Skipping.")
                else:
                    logger.warning(f"Motif {m.id} has no 'scene_objects' attribute or it's empty. Skipping for scene_object visualization.")
        else:
            logger.warning("'use_scene_objects' is True, but no motifs were provided.")
    elif motifs is not None:
        for m in motifs:
            # Ensure motifs have the necessary attributes for PlotItem
            if all(hasattr(m, attr) for attr in ['id', 'position', 'extents', 'rotation', 'object_type']):
                items_to_process.append(PlotItem(id=m.id, position=m.position, extents=m.extents, rotation=m.rotation, object_type=m.object_type))
            else:
                logger.warning(f"Motif {getattr(m, 'id', 'unknown_id')} is missing attributes. Skipping.")

    grid, door_location, bounds = create_grid(room_polygon, grid_size, door_location)

    fig, ax = plt.subplots(figsize=(20, 20))

    # Add room polygon as filled background
    x_coords, y_coords = room_polygon.exterior.xy
    ax.fill(x_coords, y_coords, alpha=0.2, fc='lightblue', label='Room Area')

    # Plot room outline
    ax.plot(x_coords, y_coords, color='red', linewidth=2)

    # Add labels for each side of the room and vertices
    vertices = list(room_polygon.exterior.coords)
    for i in range(len(vertices) - 1):
        p1 = vertices[i]
        p2 = vertices[i+1]
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        # Calculate wall angle and normal angle pointing outwards
        wall_angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        # Normal pointing left of vector p1->p2 (outwards for CCW polygon)
        normal_angle_rad = wall_angle_rad + math.pi / 2

        # Calculate label position outside the wall
        label_x = mid_x - LABEL_OFFSET * math.cos(normal_angle_rad)
        label_y = mid_y - LABEL_OFFSET * math.sin(normal_angle_rad)

        ax.text(label_x, label_y, f'wall_{i}', ha='center', va='center',
                fontsize=TEXT_SIZE,
                color='black',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

        ax.text(p1[0], p1[1] - 0.2, f'({p1[0]:.1f}, {p1[1]:.1f})',
                ha='right', va='bottom',
                fontsize=TEXT_SIZE-4,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Plot grid points
    ax.scatter(grid[:, 0], grid[:, 1], s=10, color='lightblue')

    door_angle = None
    free_space_percent = 100.0
    if door_location is not None:
        cx, cy = door_location

        min_distance = float('inf')
        closest_wall_points = None
        wall_angle_deg = 0
        for i in range(len(vertices) - 1):
            p1 = vertices[i]
            p2 = vertices[i+1]
            wall_line = LineString([p1, p2])
            distance = Point(cx, cy).distance(wall_line)
            if distance < min_distance:
                min_distance = distance
                closest_wall_points = (p1, p2)
                wall_angle_deg = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

        if closest_wall_points:
            from hsm_core.scene.geometry.grid_utils import calculate_door_angle
            door_angle = calculate_door_angle(door_location, room_polygon)

            dx_half = (DOOR_WIDTH / 2) * math.cos(math.radians(wall_angle_deg))
            dy_half = (DOOR_WIDTH / 2) * math.sin(math.radians(wall_angle_deg))
            door_p1 = (cx - dx_half, cy - dy_half)
            door_p2 = (cx + dx_half, cy + dy_half)
            ax.plot([door_p1[0], door_p2[0]], [door_p1[1], door_p2[1]],
                    color='darkgreen', linewidth=4, solid_capstyle='butt', label='Door Opening')

            pivot_x, pivot_y = door_p1
            start_angle = wall_angle_deg
            end_angle = door_angle

            sweep = (end_angle - start_angle + 360) % 360
            if sweep > 180:
                 start_angle, end_angle = end_angle, start_angle

            door_arc = patches.Arc(
                (pivot_x, pivot_y),
                DOOR_WIDTH * 2, DOOR_WIDTH * 2,
                theta1=start_angle,
                theta2=end_angle,
                color='green',
                linewidth=1.5,
                linestyle='--'
            )
            ax.add_patch(door_arc)

            dx_open = DOOR_WIDTH * math.cos(math.radians(door_angle))
            dy_open = DOOR_WIDTH * math.sin(math.radians(door_angle))
            ax.plot([pivot_x, pivot_x + dx_open],
                    [pivot_y, pivot_y + dy_open],
                    color='green', linewidth=2, linestyle='--')

            # Add Door Label near center
            ax.text(cx, cy, 'Door', ha='center', va='center', fontsize=TEXT_SIZE-2, color='darkgreen',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        else:
            logger.warning(f"Could not determine wall segment for door location {door_location}")

    if window_location is not None:
        window_width = WINDOW_WIDTH
        window_height = 0.2

        for window in window_location:
            # Find which wall the window is on by checking distance to each wall segment
            min_distance = float('inf')
            window_angle = 0
            window_point = Point(window)
            closest_wall_line = None

            # Get wall segments from room polygon
            vertices = list(room_polygon.exterior.coords)
            for i in range(len(vertices) - 1):
                p1 = vertices[i]
                p2 = vertices[i + 1]
                wall_line = LineString([p1, p2])

                distance = window_point.distance(wall_line)
                # Ensure the point is very close to the line segment
                if distance < min_distance and distance < 1e-6: # Use a small tolerance
                    min_distance = distance
                    closest_wall_line = wall_line
                    # Window should be parallel to wall
                    window_angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

            if closest_wall_line:
                # Use the original window point if it's on a wall, otherwise project
                # Projection might place it on an extension of the wall segment if not careful
                # For simplicity, let's assume the input 'window' coordinate IS the center and ON the wall
                window_x, window_y = window[0], window[1]

                # Center the rectangle at (window_x, window_y)
                # Use Affine2D for rotation around the center
                rect_center_x, rect_center_y = window_x, window_y

                # Create window rectangle patch (centered)
                window_rect = patches.Rectangle(
                    (rect_center_x - window_width / 2, rect_center_y - window_height / 2), # Bottom-left corner before rotation
                    window_width, window_height,
                    angle=0, # Rotation applied separately via transform
                    color='skyblue',
                    alpha=0.6,
                    label='Window',
                    hatch='//'
                )

                # Apply rotation transform around the center
                t = ax.transData
                rot = patches.transforms.Affine2D().rotate_deg_around(rect_center_x, rect_center_y, window_angle)
                window_rect.set_transform(rot + t)
                ax.add_patch(window_rect)

                # Add window frame (also centered and rotated)
                frame_width = window_width + 0.1
                frame_height = window_height + 0.1
                frame = patches.Rectangle(
                    (rect_center_x - frame_width / 2, rect_center_y - frame_height / 2), # Bottom-left corner before rotation
                    frame_width, frame_height,
                    angle=0, # Rotation applied separately via transform
                    color='blue',
                    fill=False,
                    linewidth=2
                )
                frame.set_transform(rot + t) # Use the same rotation transform
                ax.add_patch(frame)

                # Add label with offset based on wall angle (normal to the wall)
                label_offset = 0.3
                normal_angle_rad = math.radians(window_angle + 90)
                label_x = window_x + label_offset * math.cos(normal_angle_rad)
                label_y = window_y + label_offset * math.sin(normal_angle_rad)
            else:
                 logger.warning(f"Window location {window} not found on any wall segment.")

    # Add grid markers with coordinates if enabled
    if add_grid_markers:
        x_min, y_min, x_max, y_max = bounds
        for x_val in np.arange(x_min, x_max + grid_size, GRID_MARKER_INTERVAL):
            for y_val in np.arange(y_min, y_max + grid_size, GRID_MARKER_INTERVAL):
                point = Point(x_val, y_val)
                if room_polygon.contains(point) or room_polygon.boundary.distance(point) == 0:
                    ax.text(x_val, y_val - 0.1, f'({x_val:.1f},{y_val:.1f})',
                           color='red',
                           fontsize=TEXT_SIZE,
                           ha='center',
                           va='center',
                           bbox=dict(facecolor='white',
                                     edgecolor='none',
                                     alpha=0.7))
                    ax.plot(x_val, y_val, 'x', markersize=12, color='red')

    # Plot scene motifs with different colors
    if items_to_process: # Changed from "if motifs is not None:"
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, item in enumerate(items_to_process): # Changed from group to item, motifs to items_to_process
            color = colors[i % len(colors)]
            x_item, _, z_item = item.position # Changed from x_group, z_group, group.position

            # Ensure extents is a 3-tuple (width, height, depth)
            item_extents = item.extents
            if isinstance(item_extents, np.ndarray):
                item_extents = item_extents.tolist()
            if not (isinstance(item_extents, (list, tuple)) and len(item_extents) == 3):
                logger.warning(f"Item {item.id} has invalid extents format: {item_extents}. Skipping.")
                continue

            width, _, depth = item_extents  # Unpack (width, height, depth)
            if item.object_type == ObjectType.WALL:
                width, depth = item_extents[0], item_extents[2]

            rect = patches.Rectangle(
                (x_item - width/2, z_item - depth/2), # Used x_item, z_item
                width, depth,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.8,
                label=item.id # Used item.id
            )
            t = ax.transData
            # Convert world rotation (0°=South) to plot rotation (0°=East)
            plot_rotation = (item.rotation) % 360
            rot = patches.transforms.Affine2D().rotate_deg_around(x_item, z_item, plot_rotation) # Used x_item, z_item, plot_rotation
            rect.set_transform(rot + t)
            ax.add_patch(rect)

            arrow_length = min(width, depth) * 0.5
            theta_rad = np.radians(item.rotation)
            dx_arrow = arrow_length * np.sin(theta_rad)
            dz_arrow = arrow_length * -np.cos(theta_rad)
            ax.arrow(x_item, z_item, dx_arrow, dz_arrow, # Used x_item, z_item
                     head_width=0.1, head_length=0.1,
                     fc='black', ec='black',
                     length_includes_head=True)

            ax.text(x_item, z_item, item.id, ha='center', va='center', fontweight='bold', color='black', # Used x_item, z_item, item.id
                    fontsize=TEXT_SIZE,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Calculate free space occupancy if motifs are provided
    if items_to_process: # Changed from "if motifs is not None:"
        try:
            # Pass items_to_process to calculate_free_space_occupancy
            free_space_percent = calculate_free_space(room_polygon, items_to_process)
            ax.text(0.02, 0.98, f"Free Space: {free_space_percent:.1f}%",
                    transform=ax.transAxes,
                    fontsize=TEXT_SIZE,
                    color='darkgreen',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        except Exception as e:
            logger.error(f"Error calculating free space occupancy: {e}")

    ax.set_title(f'Room Visualization (grid size: {grid_size}m)')
    ax.set_xlabel('X-axis (width in meters)')
    ax.set_ylabel('Z-axis (length in meters)')
    ax.grid(False)

    plt.tight_layout()
    return fig, door_location, free_space_percent


def visualize_object_layout(layer_fig, layer_count, constrained_layout):
    """
    Visualize object layout across different layers.

    Args:
        layer_fig (matplotlib.figure.Figure): Figure to draw on
        layer_count (int): Number of layers to visualize
        constrained_layout (dict): Layout data with object positions and rotations

    Returns:
        matplotlib.figure.Figure: Updated figure with visualizations
    """
    # Create a new figure with subplots for each layer if not provided
    if layer_fig is None and layer_count > 0:
        layer_fig = plt.figure(figsize=(5 * layer_count, 5))

    # Create subplots for each layer
    for large_obj_name, obj_layout in constrained_layout.items():
        for layer_key, surfaces in obj_layout.items():
            if not layer_key.startswith("layer_"):
                continue

            layer_idx = int(layer_key.split("_")[1])

            # Get the corresponding subplot
            layer_axes = layer_fig.get_axes()
            if layer_idx < len(layer_axes):
                ax = layer_axes[layer_idx]
            else:
                # Create new subplot if needed
                ax = layer_fig.add_subplot(1, layer_count, layer_idx + 1)
                ax.set_title(f"Layer {layer_idx}")
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Z (meters)")
                ax.grid(True)

            # Process objects in this layer
            for surface_key, surface_objects in surfaces.items():
                # Track object positions for drawing facing lines
                object_positions = {}

                # Group objects by motif for extent visualization
                motif_groups = {}

                # draw all objects and collect positions
                for obj in surface_objects:
                    pos = np.array(obj["position"])
                    object_positions[obj["id"]] = pos

                    if "dimensions" not in obj:
                        obj["dimensions"] = [0.3, 0.3, 0.3]

                    # Add direction arrow at relative position
                    arrow_length = max(obj["dimensions"]) * 0.3
                    rotation_angle = obj["stored_rotation"] if "stored_rotation" in obj else 0

                    # For south-facing default (270 degrees), then rotate counterclockwise
                    adjusted_angle = rotation_angle - 90  # Subtract 90 to make default south
                    dx = arrow_length * np.cos(np.radians(adjusted_angle))
                    dy = arrow_length * np.sin(np.radians(adjusted_angle))
                    ax.arrow(pos[0], pos[1], dx, dy, head_width=0.02, head_length=0.02, fc="blue", ec="blue")

                    # Add object label with smaller font and compact format
                    ax.text(
                        pos[0],
                        pos[1] + max(obj["dimensions"]) * 0.5,
                        f'{obj["id"]}\n{obj["dimensions"][0]:.2f}×{obj["dimensions"][1]:.2f}m',
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                    # Group objects by motif for extent visualization
                    # Try to extract motif name from object ID (e.g., "motif_name_object_id")
                    motif_name = obj["id"].split("_")[0] if "_" in obj["id"] else obj["id"]
                    if motif_name not in motif_groups:
                        motif_groups[motif_name] = []
                    motif_groups[motif_name].append(obj)

                # Draw motif extents based on grouped objects
                for motif_name, motif_objects in motif_groups.items():
                    if len(motif_objects) > 1:  # Only show extents for motifs with multiple objects
                        # Calculate motif bounding box from object positions and dimensions
                        positions = np.array([obj["position"] for obj in motif_objects])

                        # Get dimensions for each object
                        dimensions = np.array([obj["dimensions"][:2] for obj in motif_objects])  # Only width and depth

                        # Calculate bounds
                        min_pos = positions.min(axis=0)
                        max_pos = positions.max(axis=0)

                        # Add padding based on object dimensions
                        max_dims = dimensions.max(axis=0)
                        padding = max_dims * 0.3  # 30% padding

                        # Calculate bounding box corners
                        corners = np.array([
                            [min_pos[0] - padding[0], min_pos[1] - padding[1]],  # bottom-left (X, Z)
                            [max_pos[0] + padding[0], min_pos[1] - padding[1]],  # bottom-right
                            [max_pos[0] + padding[0], max_pos[1] + padding[1]],  # top-right
                            [min_pos[0] - padding[0], max_pos[1] + padding[1]],  # top-left
                            [min_pos[0] - padding[0], min_pos[1] - padding[1]]   # close the loop
                        ])

                        # Draw motif bounding box
                        ax.plot(corners[:, 0], corners[:, 1],
                               color='red', linewidth=2, linestyle='--', alpha=0.7)

                        # Calculate center for label
                        center_x = (min_pos[0] + max_pos[0]) / 2
                        center_z = (min_pos[2] + max_pos[2]) / 2 if len(min_pos) > 2 else min_pos[1] + max_pos[1] / 2

                        # Add motif label at center
                        ax.text(center_x, center_z, f'{motif_name}',
                               ha='center', va='center', fontsize=10,
                               bbox=dict(facecolor='red', alpha=0.3, pad=2))

                # draw facing lines if applicable
                for obj in surface_objects:
                    if "stored_facing" in obj:
                        target_id = obj["stored_facing"]
                        source_pos = np.array(obj["position"])

                        if target_id in object_positions:
                            # Draw a dashed line from source to target
                            target_pos = object_positions[target_id]
                            ax.plot(
                                [source_pos[0], target_pos[0]],
                                [source_pos[1], target_pos[1]],
                                'g--',
                                linewidth=1.5,
                                alpha=0.7
                            )

                            # Add a small 'facing' label near the line midpoint
                            midpoint = (source_pos + target_pos) / 2
                            ax.text(
                                midpoint[0],
                                midpoint[1],
                                "facing",
                                ha="center",
                                va="center",
                                fontsize=6,
                                color="green",
                                bbox=dict(facecolor='white', alpha=0.5, pad=1)
                            )

            ax.set_aspect("equal")
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    layer_fig.tight_layout()
    return layer_fig