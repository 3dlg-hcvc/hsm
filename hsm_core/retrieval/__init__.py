"""
Retrieval Module for HSM
"""

from hsm_core.utils import get_logger

logger = get_logger('retrieval')

from .core.main import retrieve
from .core.adaptive_retrieval import retrieve_adaptive

from .model.model_manager import ModelManager
from .utils.transform_tracker import TransformTracker, TransformInfo

from .utils.similarities import compute_similarities
from .utils.retriever_helpers import (
    apply_hssd_alignment_transform,
    optimize_mesh_rotation,
    load_and_normalize_mesh,
    validate_support_surface_constraints,
    process_mesh_candidate,
    sort_candidates_by_quality,
    apply_mesh_to_object,
)

from .model.embeddings import load_hssd_embeddings_and_index

# Optional server imports with fallback
try:
    from .server import (
        ServerRetrievalClient,
        create_server_retrieval_client,
        RetrievalServerError,
        RetrievalServerConnectionError,
        RetrievalServerTimeoutError,
        RetrievalServerResponseError,
    )
    SERVER_AVAILABLE = True
    logger.debug("Server retrieval components available")
except ImportError as e:
    # logger.warning(f"Server retrieval components not available: {e}. Using local retrieval only.")
    SERVER_AVAILABLE = False
    class ServerRetrievalClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Server retrieval not available. Install server dependencies or use local retrieval.")

    def create_server_retrieval_client(*args, **kwargs):
        raise RuntimeError("Server retrieval not available. Install server dependencies or use local retrieval.")

    class RetrievalServerError(Exception):
        pass

    class RetrievalServerConnectionError(Exception):
        pass

    class RetrievalServerTimeoutError(Exception):
        pass

    class RetrievalServerResponseError(Exception):
        pass

__all__ = [
    "retrieve_adaptive",
    "retrieve",
    "ModelManager",
    "TransformTracker",
    "TransformInfo",
    "SERVER_AVAILABLE",
    "load_hssd_embeddings_and_index",
]

# Add server components to __all__ if available
if SERVER_AVAILABLE:
    __all__.extend([
        "ServerRetrievalClient",
        "create_server_retrieval_client",
        "RetrievalServerError",
        "RetrievalServerConnectionError",
        "RetrievalServerTimeoutError",
        "RetrievalServerResponseError",
    ])

__version__ = "1.0.0"
