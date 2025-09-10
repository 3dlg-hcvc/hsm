"""
Core retrieval functionality for HSM scene generation.
"""

from hsm_core.utils import get_logger

logger = get_logger('retrieval.core')

from .main import retrieve
from .adaptive_retrieval import retrieve_adaptive
from .retrieval_logic import run_primary_retrieval, handle_fallback_retrieval

__all__ = [
    'retrieve',
    'retrieve_adaptive',
    'run_primary_retrieval',
    'handle_fallback_retrieval',
]
