"""
Main Retrieval Module
"""

import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING
from pathlib import Path

from hsm_core.scene_motif.core.obj import Obj
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.retrieval.model.model_manager import ModelManager
from hsm_core.config import HSSD_PATH

from ..data.wn_retrieval import prepare_and_filter_candidates
from .retrieval_logic import run_primary_retrieval, handle_fallback_retrieval
from ..utils.result_handlers import apply_and_log_results
from hsm_core.utils import get_logger

logger = get_logger('retrieval.core.main')
np.random.seed(42)

from .adaptive_retrieval import SERVER_AVAILABLE
if TYPE_CHECKING and SERVER_AVAILABLE:
    from ..server import ServerRetrievalClient
else:
    class ServerRetrievalClient:
        pass
    class RetrievalServerError(Exception):
        pass
    
async def retrieve(
    objs: List[Obj],
    motif_description: str = "",
    same_per_label: bool = True,
    avoid_used: bool = False,
    randomize: bool = False,
    use_top_k: int = 5,
    force_k: int = -1,
    hssd_dir_path: Path = HSSD_PATH,
    model=None,
    max_height: float = -1.0,
    object_type: ObjectType = ObjectType.UNDEFINED,
    support_surface_constraints: Optional[Dict[str, Dict]] = None,
    server_retrieval_client: Optional[ServerRetrievalClient] = None
) -> None:
    """Batch retrieve meshes for all objects at once."""
    logger.info(f"Batch retrieving meshes for {len(objs)} objects with {object_type} type")

    try:
        # --- 1. SETUP ---
        if server_retrieval_client:
            model_instance, tokenizer = await server_retrieval_client.get_clip_model_async()
        elif model:
            model_instance, tokenizer = model
        else:
            model_instance, tokenizer = await ModelManager.get_clip_model_async()

        objs_to_process = list({obj.label: obj for obj in objs}.values()) if same_per_label else objs

        # --- 2. PREPARE & FILTER ---
        wnsynsetkeys, filtered_mesh_ids = prepare_and_filter_candidates(
            objs_to_process, object_type
        )

        # --- 3. PRIMARY RETRIEVAL ---
        obj_descriptions_list = [obj.description or obj.label for obj in objs_to_process]
        mesh_dict, used_indices = await run_primary_retrieval(
            objs_to_process, filtered_mesh_ids, obj_descriptions_list, server_retrieval_client,
            model_instance, tokenizer, hssd_dir_path,
            use_top_k, avoid_used, randomize, force_k, max_height, object_type,
            support_surface_constraints
        )

        # --- 4. FALLBACK RETRIEVAL ---
        unassigned_objs = [obj for obj in objs_to_process if obj.mesh is None]
        if unassigned_objs:
            logger.info("Starting Fallback Check for Unassigned Meshes")
            await handle_fallback_retrieval(
                unassigned_objs, wnsynsetkeys, objs_to_process, used_indices, mesh_dict,
                server_retrieval_client, model_instance, tokenizer, hssd_dir_path,
                use_top_k, avoid_used, max_height, object_type,
                support_surface_constraints, same_per_label
            )

        # --- 5. APPLY & LOG ---
        apply_and_log_results(objs, objs_to_process, mesh_dict, same_per_label)

    except Exception as e:
        logger.error(f"Error during batch retrieval: {e}")
        raise
    finally:
        if server_retrieval_client:
            await server_retrieval_client.close()
        elif model is None:
            ModelManager.clear_cache()


if __name__ == "__main__":
    from hsm_core.retrieval.data_utils import filter_hssd_categories
    categories_main = filter_hssd_categories(ObjectType.WALL)
    obj_description_example = ["lamp"]
    logger.info(f"Example filtering for WALL objects: {categories_main[:5]}")
