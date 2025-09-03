"""
Scene Motif Generation Pipeline
"""

from .processing.motif_processing import process_motif_with_visual_validation
from .llm.llm_generators import generate_arrangement_code

# Internal supporting functions
from .llm import (
    send_llm_async,
    send_llm_with_validation_async,
    send_llm_with_images_async,
    generate_arrangement
)
from .processing import (
    process_single_object_motifs,
    process_single_furniture_arrangement,
    build_arrangement_from_json
)

def batch_inference(*args, **kwargs):
    """Lazy import of batch_inference to avoid circular dependencies."""
    from .processing.batch_inference import batch_inference as _batch_inference
    return _batch_inference(*args, **kwargs)

__all__ = [
    'batch_inference',
    'process_motif_with_visual_validation',
    'generate_arrangement_code'
]
