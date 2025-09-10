"""
Retrieval Logic Module

This module contains the core retrieval logic.
"""

import random
import torch
from typing import List, Dict, Set, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
from copy import deepcopy

from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene_motif.core.obj import Obj
from hsm_core.config import HSSD_PATH
from ..utils.similarities import compute_similarities
from ..utils.retriever_helpers import process_mesh_candidate, sort_candidates_by_quality, apply_mesh_to_object
from ..utils.mesh_paths import construct_hssd_mesh_path
from ..data.data_utils import _load_hssd_alignment_data, get_fallback_mesh_ids
from hsm_core.utils import get_logger

logger = get_logger('retrieval.core.logic')

from .adaptive_retrieval import SERVER_AVAILABLE
if TYPE_CHECKING and SERVER_AVAILABLE:
    from ..server import ServerRetrievalClient
else:
    class ServerRetrievalClient:
        pass
    class RetrievalServerError(Exception):
        pass
    
async def _compute_similarities_shared(
    texts: List[str],
    filter_indices: List[str] | None,
    server_retrieval_client,
    model_instance,
    tokenizer
):
    """Shared function to compute similarities through the appropriate backend."""
    if server_retrieval_client is not None:
        # Delegate to ServerRetrievalClient
        try:
            return await server_retrieval_client.get_hssd_similarities(
                texts=texts,
                filter_indices=filter_indices,
            )
        except Exception:
            # Re-raise server errors to stop scene generation
            raise
    else:
        return compute_similarities(
            texts,
            model=model_instance,
            tokenizer=tokenizer,
            filter_indices=filter_indices or [],
        )


def _log_top_meshes_verbose(
    objs_to_process: List[Obj],
    best_similarity_indices_per_obj: List[torch.Tensor],
    final_filtered_mesh_ids_for_similarities: List[List[str]],
    similarities_list_per_obj: List[torch.Tensor],
    hssd_dir_path: Path
) -> None:
    """Log top 5 meshes for each object after computing similarities."""
    for obj_idx, obj_iter in enumerate(objs_to_process):
        logger.debug(f"Top 5 meshes for {obj_iter.label} after computing similarities:")
        if obj_idx >= len(best_similarity_indices_per_obj) or not best_similarity_indices_per_obj[obj_idx].numel() > 0:
            logger.debug("  No similar meshes found.")
            continue

        num_to_show = min(5, len(best_similarity_indices_per_obj[obj_idx]))
        for i in range(num_to_show):
            try:
                similarity_tensor_idx = best_similarity_indices_per_obj[obj_idx][i]
                file_name_mesh_id = final_filtered_mesh_ids_for_similarities[obj_idx][similarity_tensor_idx]
                mesh_path_to_print = construct_hssd_mesh_path(hssd_dir_path, file_name_mesh_id)
                logger.debug(f"  {i+1}. {mesh_path_to_print} (similarity score: {similarities_list_per_obj[obj_idx][similarity_tensor_idx]:.4f})")
            except IndexError:
                logger.debug(f"  Error accessing index {i} for {obj_iter.label}")
                break
            except Exception as e:
                logger.debug(f"  Error retrieving info for index {i} for {obj_iter.label}: {e}")


async def run_primary_retrieval(
    objs_to_process: List[Obj],
    filtered_mesh_ids: List[List[str]],
    obj_descriptions_list: List[str],
    server_retrieval_client,
    model_instance,
    tokenizer,
    hssd_dir_path: Path,
    use_top_k: int,
    avoid_used: bool,
    randomize: bool,
    force_k: int,
    max_height: float,
    object_type: ObjectType,
    support_surface_constraints: Optional[Dict[str, Dict]] = None
) -> Tuple[Dict[str, Dict], Set[str]]:
    """
    Runs the main retrieval loop using CLIP and dimension matching.

    Args:
        objs_to_process: List of objects to process
        filtered_mesh_ids: Filtered mesh IDs for each object
        obj_descriptions_list: List of object descriptions
        server_retrieval_client: ServerRetrievalClient instance or None
        model_instance: CLIP model instance
        tokenizer: CLIP tokenizer
        hssd_dir_path: Path to HSSD directory
        use_top_k: Number of top candidates to consider
        avoid_used: Whether to avoid already used meshes
        randomize: Whether to randomize selection
        force_k: Force selection of k-th candidate
        max_height: Maximum height constraint
        object_type: Type of object
        support_surface_constraints: Support surface constraints

    Returns:
        Tuple of (mesh_dict, used_indices)
    """
    if hssd_dir_path is None:
        hssd_dir_path = HSSD_PATH
        if hssd_dir_path is None:
            raise ValueError("HSSD directory path is not configured. Please check your path configuration.")

    logger.info("Computing similarities for filtered meshes")

    similarities_list_per_obj = []
    final_filtered_mesh_ids_for_similarities = []

    for obj_idx, obj_desc_iter in enumerate(obj_descriptions_list):
        current_filter_ids_for_clip = filtered_mesh_ids[obj_idx]

        if not current_filter_ids_for_clip:
            logger.warning(f"No valid mesh IDs from WN key for {obj_desc_iter}. Trying similarity without WN filtering.")

        filtered_similarities, scored_mesh_ids_for_obj = await _compute_similarities_shared(
            [obj_desc_iter],
            current_filter_ids_for_clip if current_filter_ids_for_clip else [],
            server_retrieval_client,
            model_instance,
            tokenizer
        )

        if filtered_similarities is None or filtered_similarities.shape[1] == 0:
            logger.info(f"No meshes found matching description '{obj_desc_iter}' within WN key filter.")
            similarities_list_per_obj.append(torch.tensor([], device=filtered_similarities.device if filtered_similarities is not None else 'cpu'))
            final_filtered_mesh_ids_for_similarities.append([])
        else:
            final_filtered_mesh_ids_for_similarities.append(scored_mesh_ids_for_obj)
            similarities_list_per_obj.append(filtered_similarities[0])

    # Get top K indices based on similarity scores
    best_similarity_indices_per_obj = []
    for sim_tensor in similarities_list_per_obj:
        if sim_tensor.numel() > 0:
            best_similarity_indices_per_obj.append((-sim_tensor).argsort())
        else:
            best_similarity_indices_per_obj.append(torch.tensor([], dtype=torch.long))

    _log_top_meshes_verbose(
        objs_to_process, best_similarity_indices_per_obj,
        final_filtered_mesh_ids_for_similarities, similarities_list_per_obj, hssd_dir_path
    )

    used_indices = set()
    mesh_dict = {}

    # Batch retrieve meshes for all objects
    for obj_idx, obj_iter in enumerate(objs_to_process):
        obj_iter.mesh = None
        obj_iter.mesh_path = None

        top_candidates_for_obj = []

        if obj_idx >= len(best_similarity_indices_per_obj) or not best_similarity_indices_per_obj[obj_idx].numel() > 0:
            logger.info(f"No similarity results found for object {obj_iter.label}. Will attempt fallback.")
        else:
            num_similar_for_obj = len(best_similarity_indices_per_obj[obj_idx])
            for k_rank in range(num_similar_for_obj):
                current_sim_tensor_idx = best_similarity_indices_per_obj[obj_idx][k_rank]
                mesh_id_to_load = final_filtered_mesh_ids_for_similarities[obj_idx][current_sim_tensor_idx]

                if avoid_used and mesh_id_to_load in used_indices:
                    continue

                mesh_path_to_load = construct_hssd_mesh_path(hssd_dir_path, mesh_id_to_load)

                if not mesh_path_to_load.exists():
                    continue
                candidate_result = process_mesh_candidate(
                    obj_iter,
                    mesh_path_to_load,
                    mesh_id_to_load,
                    object_type,
                    _load_hssd_alignment_data(),
                    max_height,
                    support_surface_constraints,
                )

                if candidate_result is not None:
                    top_candidates_for_obj.append(candidate_result)

                if len(top_candidates_for_obj) >= use_top_k:
                    break

        # Select final mesh for the current object from primary pass
        if not top_candidates_for_obj:
            if obj_idx < len(best_similarity_indices_per_obj) and best_similarity_indices_per_obj[obj_idx].numel() > 0:
                logger.warning(f"No loadable/optimizable meshes found for object {obj_iter.label} from top {num_similar_for_obj if 'num_similar_for_obj' in locals() else 'N/A'} candidates. Fallback will be attempted.")
        else:
            # Sort and select best candidate
            top_candidates_for_obj = sort_candidates_by_quality(top_candidates_for_obj)

            logger.debug(f"Top {len(top_candidates_for_obj)} candidates for {obj_iter.label} after sorting by (Penalized, Bounding Box Score):")
            for i, (bb_s, _, pth, m_id, rot_info, penalized_flg, constraint_info) in enumerate(top_candidates_for_obj):
                constraint_str = f", Constraints: {constraint_info}" if constraint_info else ""
                logger.debug(f"{i+1}. {Path(pth).name} (Penalized: {penalized_flg}, Score: {bb_s:.4f}, Rot: {rot_info}{constraint_str})")

            # Choose the best candidate
            choice_idx = 0
            if randomize:
                choice_idx = random.randint(0, len(top_candidates_for_obj) - 1)
            if force_k != -1:
                choice_idx = min(force_k, len(top_candidates_for_obj) - 1)

            _, selected_mesh_obj, selected_path_obj, selected_mesh_id_obj, selected_rotation_info_obj, _, _ = top_candidates_for_obj[choice_idx]

            apply_mesh_to_object(obj_iter, selected_mesh_obj, str(selected_path_obj), selected_mesh_id_obj)
            used_indices.add(selected_mesh_id_obj)

            logger.debug(f"Primary Pass: Final BBox Half Size for {obj_iter.label}: {obj_iter.bounding_box.half_size}")

            mesh_dict[obj_iter.label] = {
                "mesh": deepcopy(obj_iter.mesh),
                "path": obj_iter.mesh_path,
                "rotation_info": selected_rotation_info_obj
            }

    return mesh_dict, used_indices


async def handle_fallback_retrieval(
    unassigned_objs: List[Obj],
    wnsynsetkeys: List[Optional[str]],
    objs_to_process: List[Obj],
    used_indices: Set[str],
    mesh_dict: Dict[str, Dict],
    server_retrieval_client: ServerRetrievalClient | None,
    model_instance,
    tokenizer,
    hssd_dir_path: Path,
    use_top_k: int,
    avoid_used: bool,
    max_height: float,
    object_type: ObjectType,
    support_surface_constraints: Optional[Dict[str, Dict]] = None,
    same_per_label: bool = True
) -> None:
    """
    Attempts to find meshes for objects that failed primary retrieval.

    Args:
        unassigned_objs: List of objects without assigned meshes
        wnsynsetkeys: WordNet synset keys for objects
        objs_to_process: List of all objects being processed
        used_indices: Set of already used mesh indices
        mesh_dict: Dictionary of assigned meshes
        server_retrieval_client: ServerRetrievalClient instance or None
        model_instance: CLIP model instance
        tokenizer: CLIP tokenizer
        hssd_dir_path: Path to HSSD directory
        use_top_k: Number of top candidates to consider
        avoid_used: Whether to avoid already used meshes
        max_height: Maximum height constraint
        object_type: Type of object
        support_surface_constraints: Support surface constraints
        same_per_label: Whether to use same mesh per label
    """
    if hssd_dir_path is None:
        hssd_dir_path = HSSD_PATH
        if hssd_dir_path is None:
            raise ValueError("HSSD directory path is not configured. Please check your path configuration.")

    for obj_fb_iter in unassigned_objs:
        logger.info(f"Fallback: Object '{obj_fb_iter.label}' (desc: '{obj_fb_iter.description or obj_fb_iter.label}') has no mesh. Attempting fallback retrieval...")

        # Find the object index to get its specific wnsynsetkey
        obj_idx_for_fallback = None
        for idx, obj_temp in enumerate(objs_to_process):
            if obj_temp is obj_fb_iter:
                obj_idx_for_fallback = idx
                break

        fallback_mesh_ids_to_try = _collect_fallback_mesh_ids(
            obj_fb_iter, obj_idx_for_fallback, wnsynsetkeys,
            _load_hssd_alignment_data(), object_type
        )

        if not fallback_mesh_ids_to_try:
            logger.info(f"Fallback: No mesh IDs found for specific WN key or object type '{object_type.name}'. Cannot perform fallback for '{obj_fb_iter.label}'.")
            continue

        best_fallback_candidates = []
        for search_type, mesh_ids_set in fallback_mesh_ids_to_try:
            current_fallback_candidates = []

            logger.info(f"Fallback: {search_type} search found no similar meshes for '{obj_fb_iter.label}'. Trying direct mesh loading...")
            for mesh_id_direct in list(mesh_ids_set)[:use_top_k]:  # Try up to use_top_k meshes
                if avoid_used and mesh_id_direct in used_indices:
                    continue

                fb_mesh_path_direct = construct_hssd_mesh_path(hssd_dir_path, mesh_id_direct)
                if not fb_mesh_path_direct.exists():
                    continue

                # Process direct mesh candidate
                fb_candidate_result_direct = process_mesh_candidate(
                    obj_fb_iter,
                    fb_mesh_path_direct,
                    mesh_id_direct,
                    object_type,
                    _load_hssd_alignment_data(),
                    max_height,
                    support_surface_constraints,
                )

                if fb_candidate_result_direct is not None:
                    # Use a default CLIP score of 0.0 for direct loading
                    extended_result_direct = fb_candidate_result_direct + (0.0, search_type)
                    current_fallback_candidates.append(extended_result_direct)
                    logger.debug(f"Direct loading successful for mesh {mesh_id_direct}")

            if current_fallback_candidates:
                logger.info(f"Fallback: Found {len(current_fallback_candidates)} valid candidates from direct {search_type} loading")
                best_fallback_candidates.extend(current_fallback_candidates)
                break

        if best_fallback_candidates:
            # Sort by search type priority, then by penalized status, bbox score, and clip score
            best_fallback_candidates.sort(key=lambda x: (
                0 if x[8] == "specific_wnkey" else 1,  # search_type priority
                x[5],  # penalized flag
                x[0],  # bbox score
                -x[7]  # negative clip score for descending order
            ))

            logger.debug(f"Fallback: Top candidates for '{obj_fb_iter.label}' (Sorted by Search Type, Penalized, BBox, -CLIP Score):")
            for i, (bb_s, _, pth, m_id, r_info, pen_flg, constraint_info, cl_s, s_type) in enumerate(best_fallback_candidates[:use_top_k]):
                logger.debug(f"  {i+1}. {Path(pth).name} (Type: {s_type}, Penalized: {pen_flg}, BBox: {bb_s:.4f}, CLIP: {cl_s:.4f}, MeshID: {m_id})")

                # Select best fallback candidate
                _, fb_selected_mesh, fb_selected_path, fb_selected_mesh_id, fb_selected_rotation_info, _, _, fb_clip_score, fb_search_type = best_fallback_candidates[0]

                apply_mesh_to_object(obj_fb_iter, fb_selected_mesh, str(fb_selected_path), fb_selected_mesh_id)
                used_indices.add(fb_selected_mesh_id)

                logger.info(f"Fallback: Assigned mesh {Path(fb_selected_path).name} to '{obj_fb_iter.label}' via {fb_search_type} search. BBox: {obj_fb_iter.bounding_box.half_size}")

                # Update mesh_dict for same_per_label
                if same_per_label:
                     mesh_dict[obj_fb_iter.label] = {
                        "mesh": deepcopy(obj_fb_iter.mesh),
                        "path": obj_fb_iter.mesh_path,
                        "rotation_info": fb_selected_rotation_info
                    }
        else:
            logger.info(f"Fallback: No loadable/optimizable candidates found for '{obj_fb_iter.label}' from any fallback search type.")

    for obj_fb_iter in unassigned_objs:
        if obj_fb_iter.mesh is None:
            logger.warning(f"Unable to find mesh for '{obj_fb_iter.label}' after primary and fallback attempts.")


def _collect_fallback_mesh_ids(
    obj_fb_iter: Obj,
    obj_idx_for_fallback: Optional[int],
    wnsynsetkeys: List[Optional[str]],
    hssd_alignment_data: Dict,
    object_type: ObjectType
) -> List[Tuple[str, Set[str]]]:
    """Collect fallback mesh IDs for a specific object."""
    fallback_mesh_ids_to_try = []

    # First try: Use specific WordNet synset key IDs if available
    if (obj_idx_for_fallback is not None and
        obj_idx_for_fallback < len(wnsynsetkeys) and
        wnsynsetkeys[obj_idx_for_fallback] is not None and
        wnsynsetkeys[obj_idx_for_fallback] in hssd_alignment_data):

        specific_wn_key = wnsynsetkeys[obj_idx_for_fallback]
        specific_mesh_ids = {row["id"] for row in hssd_alignment_data[specific_wn_key]}
        fallback_mesh_ids_to_try.extend([("specific_wnkey", specific_mesh_ids)])
        logger.debug(f"Fallback: Found {len(specific_mesh_ids)} mesh IDs for specific WN key '{specific_wn_key}'")

    # Second try: Use broader object_type mesh IDs
    fallback_data = get_fallback_mesh_ids(obj_fb_iter.label, object_type)
    fallback_mesh_ids_to_try.extend(fallback_data)

    return fallback_mesh_ids_to_try
