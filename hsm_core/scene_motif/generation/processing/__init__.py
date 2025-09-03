"""
Processing Components

Main processing logic for motif generation and batch inference.
"""

from .motif_processing import process_motif_with_visual_validation, process_single_object_motifs
from .processors import process_single_furniture_arrangement, build_arrangement_from_json

# Lazy import to avoid circular dependencies
def batch_inference(*args, **kwargs):
    from .batch_inference import batch_inference as _batch_inference
    return _batch_inference(*args, **kwargs)

__all__ = [
    'batch_inference',
    'process_motif_with_visual_validation'
]
