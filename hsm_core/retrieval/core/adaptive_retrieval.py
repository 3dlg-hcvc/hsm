"""
Retrieval Factory
"""

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path

from hsm_core.scene.core.objecttype import ObjectType

if TYPE_CHECKING:
    from hsm_core.scene_motif.core.obj import Obj

from hsm_core.utils import get_logger
logger = get_logger('retrieval.core.adaptive')

try:
    from ..server import ServerRetrievalClient, RetrievalServerError
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False
    # Dummy classes for type hints
    class ServerRetrievalClient:
        pass
    class RetrievalServerError(Exception):
        pass

async def retrieve_adaptive(
    objs: List["Obj"],
    model: Any,
    motif_description: str = "",
    same_per_label: bool = True,
    avoid_used: bool = False,
    randomize: bool = False,
    use_top_k: int = 5,
    force_k: int = -1,
    hssd_dir_path: Optional[Path] = None,
    max_height: float = -1.0,
    object_type: ObjectType = ObjectType.UNDEFINED,
    support_surface_constraints: Optional[Dict[str, Dict]] = None
) -> None:
    """
    Adaptive retrieval function that automatically chooses between server and local retrieval.
    
    This function examines the model type and automatically routes to the appropriate
    retrieval implementation (server or local).
    
    Args:
        objs: List of Obj objects to retrieve meshes for
        model: Model instance (ServerRetrievalClient or local model tuple)
        motif_description: Description of the motif
        same_per_label: Whether to use same mesh per label
        avoid_used: Whether to avoid used meshes
        randomize: Whether to randomize selection
        use_top_k: Number of top candidates to consider
        force_k: Force specific candidate index
        hssd_dir_path: Path to HSSD models directory
        max_height: Maximum height constraint
        object_type: Type of objects being processed
        support_surface_constraints: Support surface constraints
    """
    from .main import retrieve

    # Determine retrieval mode
    if SERVER_AVAILABLE and isinstance(model, ServerRetrievalClient):
        server_retrieval_client = model
        local_model = None
        logger.debug("Using server retrieval")
    else:
        server_retrieval_client = None
        local_model = model
        logger.debug("Using local retrieval")

    try:
        await retrieve(
            objs=objs,
            motif_description=motif_description,
            same_per_label=same_per_label,
            avoid_used=avoid_used,
            randomize=randomize,
            use_top_k=use_top_k,
            force_k=force_k,
            hssd_dir_path=hssd_dir_path,
            model=local_model,
            max_height=max_height,
            object_type=object_type,
            support_surface_constraints=support_surface_constraints,
            server_retrieval_client=server_retrieval_client
        )
    except RetrievalServerError as e:
        # Log server error and re-raise to stop scene generation
        logger.error(f"Server retrieval failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise