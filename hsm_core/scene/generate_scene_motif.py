import os
import shutil
from pathlib import Path
from matplotlib.figure import Figure
from typing import Any, List, Dict, Union, Optional, Tuple, Hashable
from copy import deepcopy

from hsm_core.scene.motif import SceneMotif
from hsm_core.utils.plot_utils import combine_figures
from hsm_core.scene_motif.generation.processing.batch_inference import batch_inference
from hsm_core.scene.spec import ObjectSpec
from hsm_core.scene.core.objecttype import ObjectType


def _create_duplicate_files(template_motif: SceneMotif, duplicate_motif: SceneMotif, output_dir: Union[str, Path]) -> None:
    """Create directory structure and copy files for duplicate motif."""
    arrangement_dir = Path(output_dir) / Path(str(duplicate_motif.id))
    arrangement_dir.mkdir(parents=True, exist_ok=True)
    duplicate_motif.glb_path = str(arrangement_dir / f"{duplicate_motif.id}.glb")

    # Copy GLB file if it exists
    if template_motif.glb_path and Path(template_motif.glb_path).exists():
        try:
            shutil.copy2(template_motif.glb_path, duplicate_motif.glb_path)
        except Exception as e:
            print(f"Warning: Failed to copy GLB file for duplicate {duplicate_motif.id}: {e}")

    # Copy arrangement pickle file
    if template_motif.arrangement and template_motif.glb_path:
        template_pickle_path = Path(template_motif.glb_path).with_suffix('.pkl')
        duplicate_pickle_path = Path(duplicate_motif.glb_path).with_suffix('.pkl')

        if template_pickle_path.exists():
            try:
                shutil.copy2(template_pickle_path, duplicate_pickle_path)
                print(f"Copied arrangement pickle from {template_pickle_path} to {duplicate_pickle_path}")
            except Exception as e:
                print(f"Warning: Failed to copy arrangement pickle for duplicate {duplicate_motif.id}: {e}")
        else:
            # If template pickle doesn't exist, try to save arrangement directly
            try:
                if hasattr(template_motif.arrangement, 'save_pickle'):
                    template_motif.arrangement.save_pickle(str(duplicate_pickle_path))
                    print(f"Saved arrangement pickle for duplicate {duplicate_motif.id}")
            except Exception as e:
                print(f"Warning: Failed to save arrangement pickle for duplicate {duplicate_motif.id}: {e}")

async def process_scene_motifs(
    objects: List[ObjectSpec],
    motif_spec: Dict,
    output_dir: Union[str, Path],
    room_description: str,
    model: Optional[Any] = None,
    object_type: ObjectType = ObjectType.UNDEFINED,
    support_surface_constraints: Optional[Dict[str, Dict]] = None
) -> Tuple[List[SceneMotif], Optional[Figure]]:
    """
    Process scene arrangements and create scene motifs from layout data.
    
    Args:
        objects: List of ObjectSpec objects for furniture.
        motif_spec: Motif spec containing arrangements and furniture info.
        output_dir: Directory for output files.
        room_description: Description of the room.
        model: Optional CLIP model instance for inference.
        object_type: Type of objects being processed.
        support_surface_constraints: Optional constraints for small object placement on support surfaces.
        
    Returns:
        tuple: (scene_motifs, combined_fig)
            - scene_motifs: List of SceneMotif objects.
            - combined_fig: Combined matplotlib figure of motif figures or None.
    """
    scene_motifs: List[SceneMotif] = []

    furniture_lookup: Dict[int, ObjectSpec] = {item.id: item for item in objects}

    unique_arrangements: Dict[Hashable, Dict] = {}
    arrangement_groups: Dict[Hashable, List[Dict]] = {}

    for arrangement in motif_spec.get("arrangements", []):
        composition = arrangement.get("composition", {})
        furniture_entries = composition.get("furniture", [])
        furniture_signature = tuple(sorted(
            (entry["id"], entry.get("amount", 1))
            for entry in furniture_entries
        ))
        description = composition.get("description", "")
        signature = (furniture_signature, description)

        if signature not in unique_arrangements:
            unique_arrangements[signature] = arrangement
            arrangement_groups[signature] = []

        arrangement_groups[signature].append(arrangement)
    
    print(f"Found {len(unique_arrangements)} unique arrangements from {len(motif_spec.get('arrangements', []))} total arrangements")

    unique_motifs: List[SceneMotif] = []
    for signature, arrangement in unique_arrangements.items():
        composition = arrangement.get("composition", {})
        furniture_entries = composition.get("furniture", [])
        arrangement_specs = []
        for entry in furniture_entries:
            furniture_id = int(entry["id"])
            amount = int(entry.get("amount", 1))
            original_spec = furniture_lookup[furniture_id]
            furniture_spec = deepcopy(original_spec)
            furniture_spec.amount = amount
            arrangement_specs.append(furniture_spec)

        arrangement_id = arrangement.get("id", "unknown_arrangement")
        arrangement_dir = Path(output_dir) / Path(str(arrangement_id))
        arrangement_dir.mkdir(parents=True, exist_ok=True)

        motif = SceneMotif(
            id=arrangement_id,
            description=composition.get("description", ""),
            object_specs=arrangement_specs,
            extents=composition.get("total_footprint", [0,0,0]),
            height_limit=composition.get("height_limit", -1)
        )
        motif.glb_path = str(arrangement_dir / f"{arrangement_id}.glb")
        unique_motifs.append(motif)

    # Type compatibility workaround - classes are functionally equivalent
    processed_unique_motifs = await batch_inference(  # type: ignore
        unique_motifs,  # type: ignore
        Path(output_dir),
        room_description,
        model=model,
        optimize=True,
        object_type=object_type,
        support_surface_constraints=support_surface_constraints
    )

    if len(processed_unique_motifs) < 1:
        print("Warning: No motifs generated")
        return [], None
    elif len(processed_unique_motifs) < len(unique_motifs):
        print("Warning: Some unique motifs were not generated")
        missing_ids = [motif.id for motif in unique_motifs if motif not in processed_unique_motifs]
        print(f"Missing {len(unique_motifs) - len(processed_unique_motifs)} motifs: {missing_ids}")

    processed_motif_map: Dict[Hashable, SceneMotif] = {}
    for motif in processed_unique_motifs:
        for signature, arrangement in unique_arrangements.items():
            if arrangement.get("id") == motif.id:
                processed_motif_map[signature] = motif  # type: ignore
                break

    for signature, arrangement_list in arrangement_groups.items():
        if signature not in processed_motif_map:
            print(f"Warning: No processed motif found for signature {signature}")
            continue

        template_motif = processed_motif_map[signature]

        for arrangement in arrangement_list:
            if arrangement.get("id") == template_motif.id:
                scene_motifs.append(template_motif)
            else:
                duplicate_motif = deepcopy(template_motif)
                duplicate_motif.id = arrangement.get("id", f"dup_{template_motif.id}")
                _create_duplicate_files(template_motif, duplicate_motif, output_dir)
                scene_motifs.append(duplicate_motif)
    
    print(f"Generated {len(scene_motifs)} total motifs from {len(processed_unique_motifs)} unique processed motifs")

    motif_figs = [motif.fig for motif in processed_unique_motifs if hasattr(motif, 'fig') and motif.fig is not None]  # type: ignore

    if not motif_figs:
        return scene_motifs, None

    fig_height = 5 * ((len(motif_figs) + 1) // 2)

    combined_fig = combine_figures(
        figures=motif_figs,
        num_cols=2,
        figsize=(10, fig_height),
        output_path=os.path.join(output_dir, "combined_fig.png")
    )

    return scene_motifs, combined_fig
