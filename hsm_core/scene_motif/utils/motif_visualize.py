from pathlib import Path
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from typing import Dict, Optional
from matplotlib import colors as mcolors
from pathlib import Path
import os

from hsm_core.visualization.geometry_2d_visualizer import visualize_object_geometry
from hsm_core.utils.plot_utils import get_scene_transforms
from hsm_core.utils import get_logger

logger = get_logger(name='scene_motif.visualization')

def _finalize_2d_plot(ax, artists, axis_labels, glb_path, output_path, name, view_name, object_bounds=None):
    """Helper to finalize any 2D plot by fitting bounds to all artists."""
    matplotlib.use('Agg')
    
    if object_bounds is not None:
        x_min, x_max, y_min, y_max = object_bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Apply padding to prevent cropping
        padding_x = max(x_range * 0.1, 0.1)
        padding_y = max(y_range * 0.1, 0.1)
        
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
    else:
        # Fallback to matplotlib's autoscale with padding
        logger.warning(f"No object bounds provided for {view_name} plot, using autoscale fallback")
        ax.relim()
        ax.autoscale_view()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_padding = (xlim[1] - xlim[0]) * 0.2
        y_padding = (ylim[1] - ylim[0]) * 0.2
        ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)
        ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)

    plot_title = f'Objects in {name} ({view_name})' if name else f'Objects in {Path(glb_path or "scene").stem} ({view_name})'
    ax.set_title(plot_title, fontsize=12)
    ax.set_xlabel(axis_labels[0], fontsize=10)
    ax.set_ylabel(axis_labels[1], fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    
    fig = ax.get_figure()
    fig.set_size_inches(5, 5) 
    plt.tight_layout()
    
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, f"motif_view_{view_name.lower().replace(' ', '_')}.png")
        
        try:
            fig.savefig(output_file, dpi=100, bbox_inches='tight', pad_inches=0.1)
            logger.info(f"Saved {view_name} visualization to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving {view_name} visualization plot: {e}")
        finally:
            plt.close(fig)

def _visualize_scene_motif_selected_view(glb_path=None, scene=None, override_name=None, 
                                         output_path="result", verbose=True, name=None, 
                                         view_axes_indices=[0, 2], view_name="Top XZ", 
                                         axis_labels=['X-axis (width)', 'Z-axis (depth)'],
                                         draw_arrow=False, transform_matrix=[[1, 0], [0, 1]],
                                         overall_min_bounds: Optional[np.ndarray] = None,
                                         overall_max_bounds: Optional[np.ndarray] = None) -> tuple:
    """Internal helper to visualize a motif from a specified 2D view."""
    existing_objects = []
    fig, ax = plt.subplots(figsize=(5, 5))
    all_artists = [] # To store all plotted artists for accurate bounding box calculation

    # Track object bounds for fallback bounding box calculation
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')

    if glb_path is not None and not scene:
        try:
            scene = trimesh.load(glb_path, process=False)
            
            # Ensure we have a Scene object before accessing geometry
            if not isinstance(scene, trimesh.Scene):
                if isinstance(scene, trimesh.Trimesh):
                    mesh = scene
                    scene = trimesh.Scene()
                    node_name = Path(glb_path).stem
                    scene.add_geometry(mesh, node_name=node_name)
                else:
                    logger.warning(f"Loaded object is not a Trimesh Scene or Mesh: {type(scene)}")
                    return None, [], None, {}, {}, {}
        except Exception as e:
            logger.error(f"Error loading GLB file {glb_path}: {e}")
            return None, [], None, {}, {}, {}

    if scene is None or not hasattr(scene, 'geometry') or not scene.geometry:
         logger.error("Scene is empty or None")
         return None, [], None, {}, {}, {} 
         
    colors = list(mcolors.TABLEAU_COLORS.values())
    colors_names = list(mcolors.TABLEAU_COLORS.keys())
    
    transforms = get_scene_transforms(scene)
    found_valid_object = False
    
    world_meshes = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        if geometry_name not in scene.geometry:
            continue
        mesh = scene.geometry[geometry_name]
        mesh_world = mesh.copy()
        try:
            mesh_world.apply_transform(transform)
            world_meshes[node_name] = mesh_world
        except Exception as e:
            logger.warning(f"Could not apply world transform to {node_name}: {e}")

    object_index = 0
    for node_name, mesh_world in world_meshes.items():
        color = colors[object_index % len(colors)]
        color_name = colors_names[object_index % len(colors_names)]
        label = override_name[object_index] if override_name and object_index < len(override_name) else node_name
        logger.debug(f"Visualizing {view_name}: mesh {node_name} with color {color_name} and label {label}")
        
        try:
            # Use the selected view axes indices
            obj_info, processed_geometries = visualize_object_geometry(
                [mesh_world],
                ax,
                color,
                label=label,
                verbose=False,
                transform_matrix=transform_matrix,
                view_axes_indices=view_axes_indices,
                show_border=True,
                min_dimension_threshold=0.01  # Use 1cm threshold for motif visualization (vs 10cm default)
            )    
        except Exception as e:
             logger.error(f"Error during visualize_object_geometry for {label} in {view_name}: {e}")
             continue

        if obj_info:
            found_valid_object = True
            obj_data = obj_info[0] if isinstance(obj_info, list) else obj_info

            # Collect any matplotlib artists returned by visualize_object_geometry
            if processed_geometries:
                # Filter for actual matplotlib artists
                geom_artists = [geom for geom in processed_geometries if hasattr(geom, 'get_window_extent')]
                all_artists.extend(geom_artists)

            if "bounds" not in obj_data or "center" not in obj_data["bounds"]:
                 logger.warning(f"obj_info missing bounds/center for {label}")
                 continue

            center_2d = obj_data["bounds"]["center"] # Center in the selected 2D plane
            dimensions_2d = obj_data["dimensions"]   # Dimensions in the selected 2D plane
            
            bounds = obj_data["bounds"]
            
            # Update overall bounds for fallback calculation
            x_min = min(x_min, bounds["min"][0])
            x_max = max(x_max, bounds["max"][0])
            y_min = min(y_min, bounds["min"][1])
            y_max = max(y_max, bounds["max"][1])
            
            center_3d = mesh_world.centroid
            extents_3d = mesh_world.extents
            
            existing_objects.append({
                "id": object_index,
                "name": label,
                "color": color,
                "position": center_3d.tolist(),
                "dimensions": extents_3d.tolist(),
            })
            
            # Add center point marker and text label
            marker = ax.plot(center_2d[0], center_2d[1], 'rx', markersize=10)
            text_label = ax.text(center_2d[0], center_2d[1] + 0.05, f"{label}\n({center_2d[0]:.2f}, {center_2d[1]:.2f})",
                    ha='center', va='bottom', fontweight='bold', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
            all_artists.extend(marker)
            all_artists.append(text_label)
            
            # Draw orientation arrow only for top-down
            if draw_arrow and view_axes_indices == [0, 2]: # Only for top-down XZ view
                transform = transforms.get(node_name, np.eye(4)) # Get transform again
                rotation_matrix = transform[:3, :3]
                # Extract angle from rotation matrix (around Y axis for XZ plane)
                angle = np.arctan2(rotation_matrix[0, 2], rotation_matrix[0, 0]) # Check this angle calculation
                
                # Default forward direction Z- (maps to plot Y- for top-down)
                forward_dir_plot = np.array([0, -1]) 
                rot_matrix_plot = np.array([[np.cos(angle), -np.sin(angle)],
                                            [np.sin(angle), np.cos(angle)]])
                forward_dir_plot = np.dot(rot_matrix_plot, forward_dir_plot)
                
                # Draw arrow using plot coordinates (center_2d[0]=X, center_2d[1]=Z)
                arrow_length = min(dimensions_2d) * 0.5 
                head_width = min(dimensions_2d) * 0.05  
                head_length = min(dimensions_2d) * 0.1  
                if arrow_length > 1e-3:
                     arrow = ax.arrow(center_2d[0], center_2d[1],
                              forward_dir_plot[0] * arrow_length,
                              forward_dir_plot[1] * arrow_length,
                              head_width=head_width, head_length=head_length,
                              fc='black', ec='black', alpha=0.7)
                     all_artists.append(arrow)

            object_index += 1
        else:
            logger.debug(f"visualize_object_geometry returned no info for {label}")

    if not found_valid_object:
        logger.debug(f"No valid objects found to visualize in {view_name} view")
        plt.close(fig) # Close the figure if nothing was plotted
        return None, [], scene, {}, {}, {}
    
    # Prepare bounds for fallback calculation
    object_bounds = (x_min, x_max, y_min, y_max) if x_min != float('inf') else None
    
    # Finalize the plot using the helper, now passing artists and bounds
    _finalize_2d_plot(ax, all_artists, axis_labels, 
                      glb_path, output_path, name, view_name, object_bounds)
                      
    # Anchor obj logic was tied to specific views/functions and needs revisiting if required for auto-view
    # For now, just return empty dicts for hull_points and anchor_heights
    return fig, existing_objects, scene, {}, {}, {}


def visualize_scene_motif(glb_path=None, scene=None, override_name=None, 
                              output_path="result", gpt_override=None,
                              verbose=True, name=None,
                              transform_matrix=None # This will be passed as global_scene_transform
                              ) -> tuple:
    """
    Visualize all objects within a scene motif. Automatically selects the best 2D view.
    Retained for compatibility, calls visualize_scene_motif_auto_view.
    
    Args:
        glb_path (str, optional): Path to GLB file to load.
        scene (trimesh.Scene, optional): Existing scene to visualize.
        override_name (list, optional): List of names to override object labels.
        output_path (str, optional): Directory to save visualization outputs.
        gpt_override (str, optional): Override for GPT processing (unused here).
        verbose (bool, optional): Whether to print debug information.
        name (str, optional): Name for the plot title and output file.
        transform_matrix (list, optional): A 3D transformation matrix to apply to the scene if loaded from glb_path.
        
    Returns:
        tuple: (best_fig, existing_objects, scene_obj, {}, {}, None)
    """
    return visualize_scene_motif_auto_view(
        glb_path=glb_path,
        scene=scene,
        override_name=override_name,
        output_path=output_path,
        verbose=verbose,
        name=name,
        global_scene_transform=transform_matrix # Pass transform_matrix as global_scene_transform
    )

def visualize_scene_motif_auto_view(glb_path=None, scene=None, override_name=None,
                                    output_path="result", verbose=True, name=None,
                                    global_scene_transform: Optional[np.ndarray] = None) -> tuple:
    """
    Visualize the scene motif from the 2D view (Top XZ, Front XY, or Side YZ)
    that maximizes the spread of the objects' projected bounding box.

    Args:
        glb_path (str, optional): Path to GLB file to load.
        scene (trimesh.Scene, optional): Existing scene to visualize.
        override_name (list, optional): List of names to override object labels.
        output_path (str, optional): Directory to save visualization outputs.
        verbose (bool, optional): Whether to print detailed information.
        name (str, optional): Name for the plot title and output file.
        global_scene_transform (np.ndarray, optional): A 3D transformation matrix to apply to the scene
                                                       if loaded from glb_path.

    Returns:
        tuple: See _visualize_scene_motif_selected_view documentation for the chosen best view.
               Returns (None, [], None, {}, {}, {}) if loading or processing fails.
    """

    if glb_path is not None and not scene:
        try:
            scene = trimesh.load(glb_path, process=False)
            if not isinstance(scene, trimesh.Scene):
                if isinstance(scene, trimesh.Trimesh):
                    mesh = scene
                    scene = trimesh.Scene()
                    node_name = Path(glb_path).stem
                    scene.add_geometry(mesh, node_name=node_name)
                else:
                    logger.warning(f"Loaded object is not a Trimesh Scene or Mesh: {type(scene)}")
                    return None, [], None, {}, {}, {}
        except Exception as e:
            logger.error(f"Error loading GLB file {glb_path}: {e}")
            return None, [], None, {}, {}, {}

    if scene is None or not hasattr(scene, 'geometry') or not scene.geometry:
         logger.error("Scene is empty or None for auto-view calculation")
         return None, [], None, {}, {}, {} 

    # Get world-transformed meshes
    world_meshes = {}
    all_vertices_3d = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        if geometry_name not in scene.geometry:
            continue
        mesh = scene.geometry[geometry_name]
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
             continue # Skip meshes without vertices
        mesh_world = mesh.copy()
        try:
            mesh_world.apply_transform(transform)
            world_meshes[node_name] = mesh_world
            all_vertices_3d.append(mesh_world.vertices)
        except Exception as e:
            logger.warning(f"Could not apply world transform to {node_name}: {e}")

    if not all_vertices_3d:
        logger.error("No valid vertices found in any mesh")
        # Return scene in case it was loaded, but no geoms to process
        return None, [], scene if 'scene' in locals() and scene is not None else None, {}, {}, {}
        
    all_vertices_3d_np = np.concatenate(all_vertices_3d, axis=0)

    # Define candidate views
    candidate_views = {
        "Top View":   {'indices': [0, 2], 'labels': ['X-axis (width)', 'Z-axis (depth)'], 'arrow': True, "transform_matrix":[[1, 0], [0, -1]]}, # unify coord system
        "Front View": {'indices': [0, 1], 'labels': ['X-axis (width)', 'Y-axis (height)'],'arrow': False, "transform_matrix":[[1, 0], [0, 1]]},
        # "Side YZ":  {'indices': [1, 2], 'labels': ['Y-axis (height)', 'Z-axis (depth)'],'arrow': False, "transform_matrix":[[1, 0], [0, -1]]} # Y to plot-X, Z to plot-Y (flipped like Top XZ)
    }

    best_view_name = None
    
    # Only apply variance logic if there are multiple meshes with centroids
    object_centroids_3d = []
    if len(world_meshes) >= 2:
        for mesh in world_meshes.values():
            if hasattr(mesh, 'centroid') and mesh.centroid is not None:
                object_centroids_3d.append(mesh.centroid)
    
    eligible_view_names = list(candidate_views.keys()) # Default to all views

    if len(object_centroids_3d) >= 2:
        centroids_np = np.array(object_centroids_3d)
        variances = np.var(centroids_np, axis=0) # [var_x, var_y, var_z]
        max_variance = np.max(variances)

        if max_variance > 1e-6:
            primary_spread_axes_indices = {i for i, v in enumerate(variances) if np.isclose(v, max_variance)}
            
            current_eligible_views = []
            for view_name, params in candidate_views.items():
                # A view is eligible if at least one of its display axes is a primary spread axis
                if any(axis_idx in primary_spread_axes_indices for axis_idx in params['indices']):
                    current_eligible_views.append(view_name)
            
            if current_eligible_views: # If any views align with max variance
                 eligible_view_names = current_eligible_views
        # If max_variance is too small or no current_eligible_views, eligible_view_names remains all candidate_views
        
    max_spread = -1.0
    best_view_min_bounds = None
    best_view_max_bounds = None

    # Calculate spread for each eligible view
    for view_name_iter in eligible_view_names: # Renamed to avoid conflict with outer scope 'view_name'
        view_params = candidate_views[view_name_iter]
        indices = view_params['indices']
        # Project all vertices to the current view plane
        vertices_2d = all_vertices_3d_np[:, indices]
        
        current_min_bounds = None
        current_max_bounds = None
        if vertices_2d.shape[0] == 0:
             spread = 0.0
        else:
            current_min_bounds = np.min(vertices_2d, axis=0)
            current_max_bounds = np.max(vertices_2d, axis=0)
            dimensions = current_max_bounds - current_min_bounds
            # Ensure dimensions are non-negative before multiplication
            spread = max(0, dimensions[0]) * max(0, dimensions[1])
        
        logger.debug(f"View '{view_name_iter}' spread (eligible pool): {spread:.4f}")

        if spread > max_spread:
            max_spread = spread
            best_view_name = view_name_iter
            best_view_min_bounds = current_min_bounds
            best_view_max_bounds = current_max_bounds

    if best_view_name is None: # Fallback if no best view determined (e.g. all spreads zero)
        logger.warning("Could not determine best view from eligible pool based on spread. Defaulting to Top XZ")
        best_view_name = "Top XZ" # Default fallback
        if eligible_view_names and "Top XZ" not in eligible_view_names: # If Top XZ wasn't even eligible, pick first eligible
             best_view_name = eligible_view_names[0]

    logger.info(f"Selected best view: '{best_view_name}' with spread {max_spread:.4f}")

    # Get parameters for the best view
    best_view_params = candidate_views[best_view_name]

    # Call the core visualization function with the best view parameters
    best_fig, existing_objects_final, scene_final, dict1, dict2, _ = _visualize_scene_motif_selected_view(
        glb_path=None, # Pass scene directly
        scene=scene,   # Pass the prepared scene object
        override_name=override_name,
        output_path=output_path, 
        verbose=verbose,
        name=name,
        view_axes_indices=best_view_params['indices'],
        view_name=best_view_name, 
        axis_labels=best_view_params['labels'],
        draw_arrow=best_view_params['arrow'],
        transform_matrix=best_view_params['transform_matrix'],
        overall_min_bounds=best_view_min_bounds,
        overall_max_bounds=best_view_max_bounds
    )

    return best_fig, existing_objects_final, scene_final, dict1, dict2, None # Return None for the 6th element

def generate_all_motif_views(glb_path=None, scene=None, override_name=None,
                               output_path="result", verbose=True, name=None,
                               global_scene_transform: Optional[np.ndarray] = None) -> Dict[str, Figure]:
    """
    Generates and saves plots for all candidate 2D views of a scene motif.

    Args:
        glb_path (str, optional): Path to GLB file to load.
        scene (trimesh.Scene, optional): Existing scene to visualize.
        override_name (list, optional): List of names to override object labels.
        output_path (str, optional): Directory to save visualization outputs.
        verbose (bool, optional): Whether to print detailed information.
        name (str, optional): Name for the plot title and output file.
        global_scene_transform (np.ndarray, optional): A 3D transformation matrix to apply to the scene
                                                       if loaded from glb_path.

    Returns:
        Dict[str, plt.Figure]: A dictionary mapping view names to their generated matplotlib Figure objects.
                               Returns an empty dictionary if loading or processing fails.
    """
    if glb_path is not None and not scene:
        try:
            scene = trimesh.load(glb_path, process=False)
            if global_scene_transform is not None:
                logger.debug(f"Applying global_scene_transform to loaded GLB for all views: {global_scene_transform}")
                transformed_geometries = {}
                for geom_name, geom in scene.geometry.items():
                    transformed_geometries[geom_name] = geom.copy().apply_transform(global_scene_transform)
                scene.geometry = transformed_geometries
            if not isinstance(scene, trimesh.Scene):
                if isinstance(scene, trimesh.Trimesh):
                    mesh = scene
                    scene = trimesh.Scene()
                    node_name = Path(glb_path).stem
                    scene.add_geometry(mesh, node_name=node_name)
                else:
                    logger.warning(f"Loaded object is not a Trimesh Scene or Mesh: {type(scene)}")
                    return {}
        except Exception as e:
            logger.error(f"Error loading GLB file {glb_path}: {e}")
            return {}

    if scene is None or not hasattr(scene, 'geometry') or not scene.geometry:
        logger.error("Scene is empty or None")
        return {}

    all_vertices_3d = []
    for node_name_iter in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name_iter]
        if geometry_name not in scene.geometry:
            continue
        mesh = scene.geometry[geometry_name]
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            continue
        mesh_world = mesh.copy()
        try:
            mesh_world.apply_transform(transform) # Node-specific transform
            all_vertices_3d.append(mesh_world.vertices)
        except Exception as e:
            logger.warning(f"Could not apply world transform to {node_name_iter}: {e}")

    if not all_vertices_3d:
        logger.error("No valid vertices found in any mesh")
        return {}
    all_vertices_3d_np = np.concatenate(all_vertices_3d, axis=0)

    candidate_views = {
        "Top View":   {'indices': [0, 2], 'labels': ['X-axis (width)', 'Z-axis (depth)'], 'arrow': True, "transform_matrix":[[1, 0], [0, -1]]},
        "Front View": {'indices': [0, 1], 'labels': ['X-axis (width)', 'Y-axis (height)'],'arrow': False, "transform_matrix":[[1, 0], [0, 1]]},
    }

    generated_figs_dict: Dict[str, Figure] = {}
    logger.info("Generating all candidate views")

    for view_name_candidate, view_params_candidate in candidate_views.items():
        indices_candidate = view_params_candidate['indices']
        vertices_2d_candidate = all_vertices_3d_np[:, indices_candidate]

        current_min_bounds_candidate = None
        current_max_bounds_candidate = None
        if vertices_2d_candidate.shape[0] > 0:
            current_min_bounds_candidate = np.min(vertices_2d_candidate, axis=0)
            current_max_bounds_candidate = np.max(vertices_2d_candidate, axis=0)

        logger.debug(f"Generating plot for candidate view: {view_name_candidate}")

        # Call _visualize_scene_motif_selected_view to generate and save the plot
        fig_candidate, _, _, _, _, _ = _visualize_scene_motif_selected_view(
            glb_path=None, # Scene is already loaded and passed
            scene=scene,   # Pass the prepared scene object
            override_name=override_name,
            output_path=output_path,
            verbose=verbose,
            name=name,
            view_axes_indices=view_params_candidate['indices'],
            view_name=view_name_candidate, # This ensures plot is saved with correct view name in filename
            axis_labels=view_params_candidate['labels'],
            draw_arrow=view_params_candidate['arrow'],
            transform_matrix=view_params_candidate['transform_matrix'],
            overall_min_bounds=current_min_bounds_candidate,
            overall_max_bounds=current_max_bounds_candidate
        )
        if fig_candidate:
            generated_figs_dict[view_name_candidate] = fig_candidate
        else:
            logger.debug(f"No figure generated for candidate view {view_name_candidate}")
            
    return generated_figs_dict
