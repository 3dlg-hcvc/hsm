from __future__ import annotations
import math
import os
import traceback
import json
from copy import deepcopy
from pathlib import Path as PathLib

from dotenv import load_dotenv
import numpy as np
import matplotlib
from shapely.geometry import Polygon

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from hsm_core.scene.utils import *
from hsm_core.scene.validate import *
from hsm_core.support_region.analyzer import extract_support_region
from hsm_core.support_region.loader import load_support_surface
from hsm_core.scene.small_placement_util import *
from hsm_core.solvers.solver_dfs import run_solver
from hsm_core.scene.objects import LayoutData
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.utils import CompactJSONEncoder

from hsm_core.vlm.gpt import Session, extract_json
from hsm_core.vlm.utils import round_nested_values
from hsm_core.config import HSSD_PATH, PROJECT_ROOT

PROMPTS_PATH = PROJECT_ROOT / "configs" / "prompts" / "scene_prompts_small.yaml"

def get_ids_from_program_json(glb_path, large_object_names=None):
    """
    Extract object IDs and labels from program.json for multiple objects

    Args:
        glb_path: Path to GLB file
        large_object_names: List of object names to find IDs for

    Returns:
        dict: {label: id} mapping object names to their mesh IDs
    """
    program_json_path = PathLib(glb_path).parent / "program.json"

    with open(program_json_path) as f:
        data = json.load(f)

    results = {}
    for obj in data["objects"]:
        obj_str = obj.strip("Obj()")
        label = obj_str.split("label='")[1].split("'")[0]
        mesh_path = obj_str.split("mesh_path='")[1].split("'")[0]

        # Extract ID from mesh path using os.path for cross-platform compatibility
        from os.path import basename, splitext

        id = splitext(basename(mesh_path))[0]

        # Check if this label matches any of our target object names
        if large_object_names is not None:

            for target_name in large_object_names:
                if any(word in label.lower() for word in target_name.lower().split()):
                    results[label] = id
                    break
        else:
            results[label] = id

    return results

def clean_layer_info(layer_data: dict) -> dict:
    """Clean and format layer information for GPT."""
    if not layer_data:
        print("Warning: No layer data found")
        return {}
        
    structured_data = LayoutData.from_raw_layer_data(layer_data)   
    return {
        obj_name: round_nested_values(layout.to_gpt_dict())
        for obj_name, layout in structured_data.items()
    }

def collect_surface_data(
    large_object_names: list[str],
    motif: SceneMotif,
    output_dir: str,
    try_ransac: bool = True
) -> tuple[dict, dict, matplotlib.figure.Figure]:
    """
    Collect surface data for large objects including support surfaces and RANSAC data.
    
    Args:
        large_object_names: List of names of the large objects
        motif: Parent SceneMotif containing the large objects
        output_dir: Directory to save visualization outputs
        
    Returns:
        tuple: (combined_layer_data, IDs, layer_fig)
    """
    layer_fig = None
    support_scene = None
    support_data = None
    obj_layer_data = None
    combined_layer_data = {}

    # get existing objects from motif
    existing_objects, _ = motif.get_objects_by_names(large_object_names)
    IDs = {}
    for obj in existing_objects:
        try:
            IDs[obj.name] = obj.get_mesh_id()
        except Exception as e:
            print(f"Error retrieving mesh id for object {obj.name if 'name' in locals() else obj}: {e}")
    
    print(f"IDs: {IDs} with names {large_object_names}")
    
    valid_objects = []
    for large_object_name in large_object_names:
        obj_layer_data = None # Initialize here for each object
        # Check if the object name has a corresponding ID found earlier
        if large_object_name not in IDs:
            print(f"Warning: Mesh ID not found for object '{large_object_name}'. Skipping.")
            continue
            
        try:
            if IDs[large_object_name] == "e44ed1eac2b86fac8f401347a3f253edb0c87254" or IDs[large_object_name] == "9a5048c7a880859b13bfeaa9e5e1d2fab15dc592":
                print(f"Skipping {large_object_name} blacklisted")
                continue
            support_scene, support_data, image = load_support_surface(IDs[large_object_name])

            # Find matching arrangement object
            obj = None
            if motif.arrangement:
                for arrangement_obj in motif.arrangement.objs:
                    if any(word in arrangement_obj.label.lower() for word in large_object_name.lower().split()):
                        obj = arrangement_obj
                        break

            if obj is not None and obj.transform_matrix is not None:
                # Apply transform to support surface scene
                for node in support_scene.graph.nodes_geometry:
                    support_scene.graph.update(node, transform=obj.transform_matrix)
            
            layer_fig, obj_layer_data = extract_support_region(
                scene=support_scene, 
                support_data=support_data, 
                verbose=False, 
                shrink_factor=0.95
                # output_path=output_dir
            )
                
        except (FileNotFoundError, Exception) as e:
            print("Error: failed to get support surface data for ", large_object_name)
            print(traceback.format_exc())
                
        # Store layer data for this object
        if obj_layer_data is not None:
            combined_layer_data[large_object_name] = obj_layer_data
            
            # Create LayoutData from raw layer data
            layout_data_dict = LayoutData.from_raw_layer_data({large_object_name: obj_layer_data})
            
            # Find the corresponding SceneObject in existing_objects
            scene_obj = next((obj for obj in existing_objects if obj.name == large_object_name), None)
            if scene_obj:
                scene_obj.layout_data = layout_data_dict[large_object_name]
                valid_objects.append(scene_obj)  # Only add objects with valid data
            else:
                print(f"Warning: No SceneObject found for {large_object_name}")
        else:
            print(f"Warning: No layer data found for {large_object_name} during collect_surface_data")
            # continue with other objects

    # After the loop through large_object_names
    if not valid_objects:
        print("No valid objects found with surface data")
        return None, None, None
        
    return combined_layer_data, valid_objects, layer_fig

def populate_furniture(
    large_object_names: list[str],
    room_desc: str,
    generated_small_motif: list[SceneMotif],
    # motif_fig: matplotlib.figure.Figure,
    motif: SceneMotif,
    output_dir: str = "",
    target_layer: str = None
) -> tuple[dict, dict, matplotlib.figure.Figure]:
    """
    Use GPT to suggest placement of small objects on larger objects.
    
    Args:
        large_object_names: List of names of the large objects to populate
        room_desc: Description of the room
        specific_small_obj: List of SceneMotif objects representing small objects to place
        fig: Matplotlib figure showing the top-down view
        motif: Parent SceneMotif containing the large objects
        output_dir: Directory to save visualization outputs
        target_layer: Optional specific layer to target (e.g., "layer_0")
        
    Returns:
        tuple: (layout_suggestions, layer_data, layer_fig)
    """
    print(f"Populating furniture for {motif.id} with {len(generated_small_motif)} small objects")
    print(f"Specific small objects: {generated_small_motif}")
    print(f"Large object names: {large_object_names}")
    if target_layer:
        print(f"Targeting specific layer: {target_layer}")
    
    valid_ids = []
    if generated_small_motif:
        for small_motif in generated_small_motif:
            valid_ids.append(small_motif.id)

    # Collect surface data
    layer_data, _, layer_fig = collect_surface_data(large_object_names, motif, output_dir)
    if layer_data is None:
        return None, None, None

    # If we're targeting a specific layer, filter the layer_data
    if target_layer and layer_data:
        for large_obj_name in layer_data:
            filtered_layer_data = {}
            if target_layer in layer_data[large_obj_name]:
                filtered_layer_data[target_layer] = layer_data[large_obj_name][target_layer]
                # Keep non-layer keys for metadata
                for key in layer_data[large_obj_name]:
                    if not key.startswith("layer_"):
                        filtered_layer_data[key] = layer_data[large_obj_name][key]
                layer_data[large_obj_name] = filtered_layer_data
            else:
                print(f"Warning: Target layer {target_layer} not found in layer data for {large_obj_name}")

    fallback = not bool(layer_data)
    large_objects_str = ", ".join(large_object_names)
    existing_objects, _ = motif.get_objects_by_names(large_object_names)

    # images = [motif.fig]
    # images.append(layer_fig)
    
    small_obj_session = Session(PROMPTS_PATH)
    
    if fallback:
        print("Warning: Fallback to use bbox height")
        response = small_obj_session.send(
            "populate_object",
            {
                "LARGE_OBJECT": large_objects_str,
                "ROOM_TYPE": room_desc,
                "EXISTING_OBJECTS": round_nested_values(SceneObject.list_to_gpt_dict(existing_objects)),
                "OBJECT_TO_POPULATE": [
                    {
                        "id": motif.id,
                        "name": motif.object_specs[0].name if motif.object_specs else "",
                        "description": motif.description
                    } for motif in generated_small_motif
                ] if generated_small_motif else "None",
                "TARGET_LAYER": target_layer or "any"  # Pass the target layer to GPT
            },
            images=[motif.fig],
            json=True,
            verbose=True,
        )
    else:
        # Add layer targeting information to the prompt
        layer_prompt = f" specifically on the {target_layer} layer" if target_layer else ""
        small_obj_session.send(
            "describe_layered_object",
            {
                "LARGE_OBJECT": large_objects_str,
                "ROOM_TYPE": room_desc,
                "EXISTING_OBJECTS": round_nested_values(SceneObject.list_to_gpt_dict(existing_objects)),
                "OBJECT_TO_POPULATE": [
                    {
                        "id": motif.id,
                        "description": motif.description
                    } for motif in generated_small_motif
                ] if generated_small_motif else f"*Suggest small objects to populate{layer_prompt}*",
            },
            images=layer_fig,
            verbose=True,
        )
        _, motif_obj_names = motif.get_objects_by_names(large_object_names)
        # Clean layer data before sending to GPT
        cleaned_layer_data = clean_layer_info(layer_data)
        
        # Get all object names in the motif for facing validation (not just large_object_names)
        all_motif_objects, parent_motif_obj_names = motif.get_objects_by_names()
        
        response = small_obj_session.send_with_validation(
            "populate_object_layered",
            {
                "LARGE_OBJECT": large_objects_str,
                "ROOM_TYPE": room_desc,
                "PARENT_MOTIF_DESCRIPTION": motif.description,
                "SMALL_MOTIFS_TO_POPULATE": [
                    {
                        "id": small_motif.id, 
                        "description": small_motif.description,
                        "size": f"{small_motif.extents[0]:.2f}m × {small_motif.extents[2]:.2f}m"
                    } for small_motif in generated_small_motif
                ] if generated_small_motif else "Suggest appropriate small objects for this context",
                "PARENT_MOTIF_OBJECTS": round_nested_values(SceneObject.list_to_gpt_dict(existing_objects)),
                "LAYER_INFO": round_nested_values(cleaned_layer_data)
            },
            lambda response: validate_layered_layout(response, 
                                                     layer_data, 
                                                     large_object_names,
                                                     list(valid_ids), 
                                                     target_layer, 
                                                     parent_motif_obj_names),
            json=True,
            verbose=True,
            images=motif.fig
        )
    json_data = json.loads(extract_json(response))
    
    # If we're targeting a specific layer, ensure the response only contains that layer
    if target_layer and json_data:
        for large_obj_name in json_data:
            filtered_json_data = {}
            if target_layer in json_data[large_obj_name]:
                filtered_json_data[target_layer] = json_data[large_obj_name][target_layer]
                # Keep non-layer keys for metadata
                for key in json_data[large_obj_name]:
                    if not key.startswith("layer_"):
                        filtered_json_data[key] = json_data[large_obj_name][key]
                json_data[large_obj_name] = filtered_json_data
    
    return json_data, layer_data, layer_fig

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

def optimize_small_objects(
    cfg,
    fig,
    layout_suggestions,
    output_path="result",
    subfix="",
    layer_fig=None,
    layer_data: dict = {},
    verbose=True,
    small_motifs: List[SceneMotif] = None,
    fallback=False
) -> dict:
    """
    Visualize small object placement within a scene, handling multiple layers and surfaces.

    Args:
        fig: Matplotlib figure for visualization
        layout_suggestions: Layout data with object positions
        output_path: Directory to save visualization outputs
        subfix: Suffix for output filenames
        layer_fig: Optional pre-existing layer figure
        layer_data: Layer information for surfaces
        verbose: Whether to print detailed information
        small_motifs: List of SceneMotif objects representing small objects to place

    Returns:
        dict: Occupancy data containing:
        - 'average_occupancy': weighted average occupancy across all surfaces
        - 'surfaces_processed': number of surfaces processed
        - 'total_occupancy': sum of all surface occupancies
    """
    if verbose:
        print("\nStarting visualization with small objects...")
        print(f"Layout suggestions: {json.dumps(layout_suggestions, cls=CompactJSONEncoder, indent=2)}\n")
        print(f"Layer data: {json.dumps(layer_data, cls=CompactJSONEncoder, indent=2)}\n")

    constrained_layout = deepcopy(layout_suggestions)

    # Track occupancy data across all surfaces
    total_occupancy = 0.0
    surfaces_processed = 0

    # Process each large object and its layers/surfaces
    for large_obj_name, obj_layout in constrained_layout.items():
        for layer_key, surfaces in obj_layout.items():
            if not layer_key.startswith("layer_"):
                continue

            # Get layer information including surface geometries
            # Extract base name from instance name (e.g., "nightstand_1" -> "nightstand")
            base_obj_name = large_obj_name.split('_')[0] if '_' in large_obj_name and large_obj_name.split('_')[-1].isdigit() else large_obj_name
            
            # Try instance name first, then base name
            layer_info = layer_data.get(large_obj_name, {}).get(layer_key, {})
            if not layer_info:
                layer_info = layer_data.get(base_obj_name, {}).get(layer_key, {})
            
            if not layer_info:
                print(f"Warning: No layer info found for {layer_key} (tried {large_obj_name} and {base_obj_name})")
                continue

            # Process each surface in the layer
            for surface_key, surface_objects in surfaces.items():
                print(f"Constraining objects in {large_obj_name} - {layer_key} - {surface_key}")
                surface_id = int(surface_key.split("_")[1])
                surface_info = next((s for s in layer_info.get("surfaces", []) if s["surface_id"] == surface_id), None)

                if not surface_info:
                    print(f"Warning: No surface info found for {surface_key} (ID: {surface_id})")
                    continue
                if "geometry" not in surface_info:
                    print(f"Warning: No geometry data found for {surface_key} (ID: {surface_id}), skipping surface")
                    continue

                # Format surface objects using small_motifs information
                formatted_objects = []
                for obj in surface_objects:
                    # Find matching motif
                    matching_motif = next((motif for motif in small_motifs if motif.id == obj["id"]), None) if small_motifs else None
                    
                    if matching_motif:
                        # Create formatted object with motif information
                        # Ensure dimensions are properly converted to individual float values
                        dimensions = [0.1,0.1,0.1]
                        if hasattr(matching_motif, 'extents') and matching_motif.extents is not None:
                            # Convert extents to list of floats, handling nested structures
                            extents = matching_motif.extents
                            if isinstance(extents, (list, tuple)) and len(extents) >= 3:
                                dimensions = [float(extents[0]), float(extents[1]), float(extents[2])]
                        
                        formatted_obj = {
                            "id": obj["id"],
                            "name": obj["id"],
                            "dimensions": dimensions,
                            "position": obj["position"],
                        }
                        
                        # # Handle rotation - both angle and facing cases
                        rotation_value = obj["rotation"]
                        if isinstance(rotation_value, dict):
                            if "angle" in rotation_value:
                                # Direct angle rotation
                                formatted_obj["rotation"] = float(rotation_value["angle"])
                                formatted_obj["stored_rotation"] = float(rotation_value["angle"])
                            elif "facing" in rotation_value:
                                # Store the facing target and use default rotation for now
                                # The actual facing angle will be calculated in update_small_motifs_from_constrained_layout
                                formatted_obj["stored_facing"] = rotation_value["facing"]
                                formatted_obj["rotation"] = 0  # Default rotation until facing is processed
                                formatted_obj["stored_rotation"] = 0
                            elif "face_away" in rotation_value:
                                # Store the face_away target and use default rotation for now
                                # The actual facing angle will be calculated in update_small_motifs_from_constrained_layout
                                formatted_obj["stored_face_away"] = rotation_value["face_away"]
                                formatted_obj["rotation"] = 0  # Default rotation until face_away is processed
                                formatted_obj["stored_rotation"] = 0
                        else:
                            print(f"Warning: Invalid rotation value for {obj['id']}: {rotation_value}")
                            formatted_obj["rotation"] = 0
                            formatted_obj["stored_rotation"] = 0
                            
                        formatted_objects.append(formatted_obj)
                    else:
                        print(f"Warning: No matching motif found for object {obj['id']}")
                        # Convert legacy 2D position to 3D
                        obj_copy = obj.copy()
                        if "position" in obj_copy and len(obj_copy["position"]) == 2:
                            obj_copy["position"] = [obj_copy["position"][0], 0.0, obj_copy["position"][1]]
                        formatted_objects.append(obj_copy)

                # Solve collisions for objects on this surface
                print(f"formatted_objects: {formatted_objects}")
                
                # Skip solver if no objects to place
                if not formatted_objects:
                    print(f"No objects to place on {surface_key}, skipping solver")
                    solved_objects_on_surface = []
                else:
                    surface_geometry = Polygon(surface_info["geometry"]["vertices"])
                    solved_objects_on_surface, surface_occupancy = run_solver(
                        surface_motifs=formatted_objects,
                        surface_geometry=surface_geometry,
                        grid_size=0.01,
                        expand_extent=1.05,
                        output_dir=output_path,
                        subfix=f"{surface_key}",
                        fallback=fallback,
                        verbose=False,
                        enable=cfg.mode.use_solver
                    )

                    # Accumulate occupancy data
                    total_occupancy += surface_occupancy
                    surfaces_processed += 1

                    if solved_objects_on_surface:
                        solved_ids = {m["id"] for m in solved_objects_on_surface}
                        formatted_objects = [m for m in formatted_objects if m["id"] in solved_ids]

                # Restore facing information to solved objects
                # Create a mapping from object IDs to their facing information
                facing_lookup = {}
                for original_obj in formatted_objects:
                    obj_id = original_obj["id"]
                    facing_data = {}
                    if "stored_facing" in original_obj:
                        facing_data["stored_facing"] = original_obj["stored_facing"]
                    if "stored_face_away" in original_obj:
                        facing_data["stored_face_away"] = original_obj["stored_face_away"]
                    if "stored_rotation" in original_obj:
                        facing_data["stored_rotation"] = original_obj["stored_rotation"]
                    if facing_data:  # Only store if there's facing information
                        facing_lookup[obj_id] = facing_data

                # Apply facing information to solved objects
                for solved_obj in solved_objects_on_surface:
                    obj_id = solved_obj["id"]
                    if obj_id in facing_lookup:
                        solved_obj.update(facing_lookup[obj_id])

                constrained_layout[large_obj_name][layer_key][surface_key] = solved_objects_on_surface
    
    # Calculate total number of layers
    layer_count = 0
    for obj_layout in layout_suggestions.values():
        layer_keys = [k for k in obj_layout.keys() if k.startswith("layer_")]
        if layer_keys:
            max_layer = max(int(k.split("_")[1]) for k in layer_keys)
            layer_count = max(layer_count, max_layer + 1)

    # Visualize the layout
    layer_fig = visualize_object_layout(layer_fig, layer_count, constrained_layout)

    # Save figure
    output_file = os.path.join(output_path, f"small_obj_layout{subfix}.png")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    layer_fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved small object layout visualization to: {output_file}")
    plt.close(layer_fig)

    # Calculate occupancy statistics
    average_occupancy = total_occupancy / surfaces_processed if surfaces_processed > 0 else 0.0

    occupancy_data = {
        'average_occupancy': average_occupancy,
        'surfaces_processed': surfaces_processed,
        'total_occupancy': total_occupancy
    }

    return occupancy_data, constrained_layout

def _local_to_world(
    parent_pos: Tuple[float, float, float],
    parent_rot_deg: float,
    surface_offset: Tuple[float, float, float],
    local_uv: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Convert a 2-D solver coordinate (u, v) expressed in the local
    X–Z plane of a *surface* into room/world coordinates.

    Args:
        parent_pos:     The parent object's world position (its *local* origin).
        parent_rot_deg: Parent yaw in degrees (counter-clockwise, 0° = +Z).
        surface_offset: (x, y, z) position of the **surface centre** in the
                       *parent's* local frame.  `y` is the layer height.
        local_uv:       2-tuple `(u, v)` returned by the DFS solver where
                       • `u` is the offset along the surface's local X axis.
                       • `v` is the offset along the surface's local Z axis
                         (depth / forward direction).

    Returns:
        A 3-tuple giving the small object's centre in world space.
    """
    # Build local 3-D vector inside the parent frame
    local_vector = np.array(
        [surface_offset[0] + local_uv[0],  # X
         surface_offset[1],               # Y (layer height)
         surface_offset[2] + local_uv[1]] # Z
    )

    # Parent to world rotation
    theta = math.radians(parent_rot_deg % 360)
    rot_mat = np.array(
        [[-math.cos(theta), 0.0, -math.sin(theta)],
         [0.0,              1.0,  0.0            ],
         [ math.sin(theta), 0.0, -math.cos(theta)]]
    )

    world_offset = rot_mat @ local_vector
    return tuple(np.array(parent_pos) + world_offset)

def update_small_motifs_from_constrained_layout(
    constrained_layout: Dict[str, Any],
    small_motifs: List[SceneMotif],
    parent_motif: SceneMotif,
    layer_data: dict,
) -> List[SceneMotif]:
    """
    Update small SceneMotif objects with world coordinates based on a constrained layout.
    
    Args:
        constrained_layout (dict): Layout data from collision solving (keyed by large object names,
            containing layer and surface details with small motif positions and rotations).
        small_motifs (List[SceneMotif]): List of small SceneMotif objects to update.
        parent_motif (SceneMotif): The parent SceneMotif containing the large objects.
        layer_data (dict): Dictionary containing layer information (including 'height') per large object per layer.
        raycast_enabled (bool): Whether to enable raycasting for layer height refinement.
        
    Returns:
        List[SceneMotif]: The updated small SceneMotif objects with new positions and rotations.
    """
    updated_motifs: List[SceneMotif] = []
    print(f"Updating small motifs from constrained layout: {constrained_layout}")
    
    # Track counts for each small_id to handle duplicates by appending a suffix.
    id_counts: Dict[str, int] = {}
    
    # Build a lookup dictionary of all available objects in the parent motif
    # This will be used for "facing" rotations
    object_lookup = {}
    for obj in parent_motif.objects:
        object_lookup[obj.name] = obj
        # Also add with lowercase for more flexible lookups
        object_lookup[obj.name.lower()] = obj
        # If the object has an ID, also add it as a key
        if hasattr(obj, 'id') and obj.id:
            object_lookup[obj.id] = obj
    
    # Also add the small motifs to the lookup
    for motif in small_motifs:
        object_lookup[motif.id] = motif
        if hasattr(motif, 'name') and motif.name:
            object_lookup[motif.name] = motif
    
    # Iterate over each large object in the constrained layout.
    for large_obj_name, obj_layout in constrained_layout.items():
        # Find the corresponding parent SceneObject within the parent motif.
        parent_obj = next(
            (obj for obj in parent_motif.objects if large_obj_name.lower() in obj.name.lower()),
            None
        )
        if not parent_obj:
            print(f"Warning: Parent object for '{large_obj_name}' not found in parent motif '{parent_motif.id}'.")
            continue
        
        # Use optimized world position if available (from spatial optimization)
        if hasattr(parent_obj, 'get_world_position'):
            parent_pos = parent_obj.get_world_position()
        elif hasattr(parent_obj, 'optimized_world_pos') and parent_obj.optimized_world_pos is not None:
            parent_pos = parent_obj.optimized_world_pos
        else:
            parent_pos = parent_obj.position
        
        try:
            parent_angle = float(parent_obj.rotation)
        except (TypeError, ValueError) as e:
            print(f"Error: Invalid rotation for parent object '{parent_obj.name}': {e}")
            continue
        # parent_rad = math.radians(parent_angle) # Not directly used before local_pos processing
        
        # Get parent front vector if available
        parent_front_vector = parent_obj.front_vector
        
        # Iterate over the layer entries for this large object.
        for layer_key, surfaces in obj_layout.items():
            if not layer_key.startswith("layer_"):
                continue
            
            # Determine the layer height using layer_data.
            # Extract base name from instance name (e.g., "nightstand_1" -> "nightstand")
            base_obj_name = large_obj_name.split('_')[0] if '_' in large_obj_name and large_obj_name.split('_')[-1].isdigit() else large_obj_name
            
            # Try instance name first, then base name
            layer_info = layer_data.get(large_obj_name, {}).get(layer_key, {})
            if not layer_info:
                layer_info = layer_data.get(base_obj_name, {}).get(layer_key, {})
            
            # Determine the Y offset from the parent object's y-origin (parent_pos[1]).
            parent_y_origin: float
            if isinstance(layer_info, dict) and "height" in layer_info:
                parent_y_origin = layer_info["height"]
            else:
                parent_y_origin = 0.0
            
            # Calculate an initial Y position for the small object's base on the parent surface.
            initial_layer_height: float = parent_pos[1] + parent_y_origin

            # Iterate over each surface in the layer.
            for surface_key, objects_on_surface in surfaces.items():
                # --- Refine layer_height once per specific surface via raycasting ---
                # Start with the layer's initial height, then refine for this specific surface if raycast is enabled.
                refined_layer_height = initial_layer_height

                # Iterate over each object data on this surface.
                for obj_data in objects_on_surface:
                    small_id = obj_data["id"]
                    current_count = id_counts.get(small_id, 0)
                    id_counts[small_id] = current_count + 1

                    small_motif_template = next((m for m in small_motifs if m.id == small_id), None)
                    if not small_motif_template:
                        print(f"Warning: Small motif with id '{small_id}' not found.")
                        continue
                    
                    # Create a deep copy of the template motif for this specific instance
                    small_motif = deepcopy(small_motif_template)
                    pos_from_solver = obj_data["position"] # This is local to the surface plane (e.g., [x,z] or [x,0,z])

                    local_x, local_z = 0.0, 0.0 # Initialize fallback values

                    # Directly use solver's output X and Z as local_x and local_z
                    # relative to the parent object's origin.
                    # pos_from_solver can be [x, z] or [x, 0.0, z] from the run_solver output.
                    if pos_from_solver and isinstance(pos_from_solver, list):
                        if len(pos_from_solver) == 2:
                            # Format: [x, z] - most common case
                            local_x = pos_from_solver[0]
                            local_z = pos_from_solver[1]
                        elif len(pos_from_solver) == 3:
                            # Format: [x, y, z] - use x and z, ignore y
                            local_x = pos_from_solver[0]
                            local_z = pos_from_solver[2]
                        else:
                            print(f"Warning: Unexpected pos_from_solver length {len(pos_from_solver)} for {small_id}. Using (0,0) for local_x, local_z.")
                            local_x, local_z = 0.0, 0.0
                        
                        # For logging/debugging, we can note how local_x, local_z were derived.
                        obj_data["derived_local_xz_for_world_transform"] = [local_x, local_z]
                        obj_data["transformed_local_position_to_parent"] = f"Directly from solver: x={local_x:.4f}, z={local_z:.4f}" # Adjusted debug message
                    else:
                        print(f"Warning: Unexpected pos_from_solver format {pos_from_solver} for {small_id}. Using (0,0) for local_x, local_z.")
                        local_x, local_z = 0.0, 0.0 # Fallback
                        obj_data["derived_local_xz_for_world_transform"] = "Error: Bad solver pos format or missing"
                        obj_data["transformed_local_position_to_parent"] = "Error: Bad solver pos format, using fallback"

                    # local_x, local_z now correctly refer to coordinates in the parent object's local frame
                    
                    parent_rad = math.radians(parent_angle)

                    # --------------------------------------------------------------
                    # Compute world-space coordinates using unified helper.
                    # We still need a *semantic* distinction for WALL parents:
                    # the solver returns (x, height_offset) instead of depth.
                    # --------------------------------------------------------------

                    is_wall_parent = hasattr(parent_obj, "obj_type") and parent_obj.obj_type == ObjectType.WALL

                    height_offset = 0.0005
                    if is_wall_parent and pos_from_solver and isinstance(pos_from_solver, list) and len(pos_from_solver) >= 2:
                        # height_offset = pos_from_solver[1]
                        local_x = pos_from_solver[0]

                        # Depth so that the small object sits in front of the wall.
                        parent_depth = parent_obj.dimensions[2] if len(parent_obj.dimensions) > 2 else 0.2
                        small_depth = (
                            small_motif.extents[2]
                            if hasattr(small_motif, "extents") and small_motif.extents is not None and len(small_motif.extents) > 2
                            else 0.1
                        )
                        # local_z = parent_depth / 2 + small_depth / 2
                        local_z = pos_from_solver[1]

                        # # If the wall parent faces 180°, flip X to preserve left/right ordering.
                        # if is_wall_parent and int(round(parent_angle)) % 360 == 180:
                        #     local_x *= -1.0

                    # Retrieve surface metadata (centre + layer height)
                    parent_key = large_obj_name if large_obj_name in layer_data else base_obj_name
                    surface_list = layer_data.get(parent_key, {}).get(layer_key, {}).get("surfaces", [])
                    surface_id = int(surface_key.split("_")[-1]) if surface_key.startswith("surface_") else None
                    surface_info = next((s for s in surface_list if s.get("surface_id") == surface_id), None)

                    if is_wall_parent and surface_info is not None:
                        # Use the unified helper for wall parents
                        center_x, center_z = surface_info.get("center", [0.0, 0.0])
                        layer_height = parent_y_origin  # y of surface in parent frame

                        world_pos = _local_to_world(
                            parent_pos=parent_pos,
                            parent_rot_deg=parent_angle,
                            surface_offset=(center_x, layer_height + height_offset, center_z),
                            local_uv=(local_x, local_z),
                        )
                    else:
                        # Fallback (and the default path for non-wall parents): original rotation math
                        if surface_info is None and is_wall_parent:
                            print(
                                f"Warning: Could not locate surface info for {parent_key} {layer_key} {surface_key}; "
                                "using trigonometric fallback."
                            )

                        parent_rad = math.radians(parent_angle)
                        world_x = parent_pos[0] + (local_x * math.cos(parent_rad) - local_z * math.sin(parent_rad))
                        world_z = parent_pos[2] + (local_x * math.sin(parent_rad) + local_z * math.cos(parent_rad))
                        world_y = parent_pos[1] + parent_y_origin + height_offset
                        world_pos = (world_x, world_y, world_z)

                    # Ensure object's bottom is not below the parent's surface
                    if is_wall_parent:
                        parent_surface_y = parent_pos[1] + parent_y_origin
                        min_allowed_y_center = parent_surface_y  + 0.001 # 1mm clearance
                        
                        if world_pos[1] < min_allowed_y_center:
                            print(f"  Clamping Y position for {small_id}: {world_pos[1]:.3f}m -> {min_allowed_y_center:.3f}m")
                            world_pos = (world_pos[0], min_allowed_y_center, world_pos[2])

                    # Process rotation - handle both angle and facing cases
                    rotation_data = obj_data.get("rotation", 0)
                    local_rotation = 0
                    facing_target = None
                    face_away_target = None
                    
                    # Check for stored facing information first (preserved from solver)
                    if "stored_facing" in obj_data:
                        facing_target = obj_data["stored_facing"]
                        print(f"  - Object {small_id} set to face target: {facing_target} (from stored_facing)")
                    elif "stored_face_away" in obj_data:
                        face_away_target = obj_data["stored_face_away"]
                        print(f"  - Object {small_id} set to face away from target: {face_away_target} (from stored_face_away)")
                    elif isinstance(rotation_data, dict):
                        if "angle" in rotation_data:
                            try:
                                local_rotation = float(rotation_data["angle"])
                            except (TypeError, ValueError):
                                print(f"Warning: Invalid rotation angle for {small_id}: {rotation_data['angle']}")
                                local_rotation = 0
                        elif "facing" in rotation_data:
                            facing_target = rotation_data["facing"]
                            print(f"  - Object {small_id} set to face target: {facing_target} (from rotation_data)")
                        elif "face_away" in rotation_data:
                            face_away_target = rotation_data["face_away"]
                            print(f"  - Object {small_id} set to face away from target: {face_away_target} (from rotation_data)")
                    else:
                        # Handle stored_rotation if available, otherwise use rotation_data directly
                        if "stored_rotation" in obj_data:
                            try:
                                local_rotation = float(obj_data["stored_rotation"])
                            except (TypeError, ValueError):
                                print(f"Warning: Invalid stored_rotation for {small_id}: {obj_data['stored_rotation']}")
                                local_rotation = 0
                        else:
                            try:
                                local_rotation = float(rotation_data)
                            except (TypeError, ValueError):
                                print(f"Warning: Invalid rotation for {small_id}: {rotation_data}")
                                local_rotation = 0
                    
                    # Determine the base orientation of the parent object
                    base_parent_orientation_deg: float
                    # parent_front_vector is parent_obj.front_vector, parent_angle is float(parent_obj.rotation)
                    if parent_front_vector is not None and \
                       (not isinstance(parent_front_vector, (float, int))) and \
                       len(parent_front_vector) >= 3 and \
                       (parent_front_vector[0] != 0.0 or parent_front_vector[2] != 0.0):
                        # Use the front_vector to determine parent's orientation in XZ plane.
                        # relative to +Z axis, with +X to the right.
                        base_parent_orientation_deg = math.degrees(math.atan2(parent_front_vector[0], parent_front_vector[2]))
                    else:
                        # Fallback to the raw parent_angle if front_vector is not usable (None, or [0,Y,0]).
                        base_parent_orientation_deg = parent_angle
                    
                    # Default world rotation calculation (if not facing another object)
                    world_rotation = (base_parent_orientation_deg + local_rotation) % 360
                    
                    # Handle facing target
                    if facing_target:
                        # Look up the target object
                        target_obj = None
                        if facing_target in object_lookup:
                            target_obj = object_lookup[facing_target]
                        elif facing_target.lower() in object_lookup:
                            target_obj = object_lookup[facing_target.lower()]
                        
                        if target_obj and hasattr(target_obj, 'position'):
                            target_pos = target_obj.position
                            # Calculate direction vector from small object to target
                            direction_x = target_pos[0] - world_pos[0]
                            direction_z = target_pos[2] - world_pos[2]
                            
                            # Calculate angle in degrees (0 is positive z-axis, increases clockwise)
                            facing_angle = math.degrees(math.atan2(direction_x, direction_z))
                            # Use the facing angle directly since 0° already points in the correct direction
                            world_rotation = facing_angle % 360
                            print(f"  - Calculated facing angle to {facing_target}: {facing_angle}° (world rotation: {world_rotation}°)")
                        else:
                            print(f"Warning: Could not find target object '{facing_target}' for facing rotation")
                    
                    # Handle face_away target
                    elif face_away_target:
                        # Look up the target object
                        target_obj = None
                        if face_away_target in object_lookup:
                            target_obj = object_lookup[face_away_target]
                        elif face_away_target.lower() in object_lookup:
                            target_obj = object_lookup[face_away_target.lower()]
                        
                        if target_obj and hasattr(target_obj, 'position'):
                            target_pos = target_obj.position
                            # Calculate direction vector from small object to target
                            direction_x = target_pos[0] - world_pos[0]
                            direction_z = target_pos[2] - world_pos[2]
                            
                            # Calculate angle pointing toward target, then add 180° to face away
                            toward_angle = math.degrees(math.atan2(direction_x, direction_z))
                            face_away_angle = (toward_angle + 180) % 360
                            world_rotation = face_away_angle
                            print(f"  - Calculated face_away angle from {face_away_target}: {face_away_angle}° (toward: {toward_angle}°)")
                        else:
                            print(f"Warning: Could not find target object '{face_away_target}' for face_away rotation")
                    
                    # Update the motif with computed world position and rotation.
                    small_motif.position = world_pos
                    small_motif.rotation = world_rotation
                    small_motif.object_type = ObjectType.SMALL
                    small_motif.placement_data = {
                        "parent_object": large_obj_name,
                        "layer": layer_key,
                        "surface": surface_key,
                        "local_position": pos_from_solver,
                        "local_rotation": local_rotation,
                        "computed_world_position": world_pos,
                        "computed_world_rotation": world_rotation,
                        "debug_surface_y": refined_layer_height,
                        # "debug_small_obj_half_height": small_obj_half_height
                    }
                    
                    # Add facing information if applicable
                    if facing_target:
                        small_motif.placement_data["facing_target"] = facing_target
                    elif face_away_target:
                        small_motif.placement_data["face_away_target"] = face_away_target
                    
                    updated_motifs.append(small_motif)
                    
                    # Print detailed debugging info for this small object
                    print(f"DEBUG: Placed small object '{small_motif.id}' on parent '{large_obj_name}'")
                    print(f"  - Parent position: {parent_pos}")
                    print(f"  - Parent rotation: {parent_angle}°")
                    print(f"  - Local position: {pos_from_solver}")
                    print(f"  - Local rotation: {local_rotation}°")
                    print(f"  - World position: {world_pos}")
                    print(f"  - World rotation: {world_rotation}°")
                    if facing_target:
                        print(f"  - Facing target: {facing_target}")
                    print(f"  - Layer: {layer_key}, Surface: {surface_key}")
    return updated_motifs
