"""
Motif Utilities

Shared utilities for motif visualization, persistence, validation, and processing.
"""

from .motif_visualize import (
    visualize_scene_motif,
    visualize_scene_motif_auto_view,
    generate_all_motif_views
)
from .mesh_utils import (
    create_furniture_lookup,
    assign_mesh_to_object,
    assign_mesh_to_object_with_normalization
)
from .utils import (
    log_time,
    calculate_arrangement_half_size,
    extract_objects,
    resolve_sub_arrangements,
    persist_motif_arrangement,
)
from .saving import save_arrangement
from .validation import (
    validate_remaining_arrangements,
    validate_compositional_json,
    is_sm_exceeds_support_region,
    inference_validation,
)
from .logger import MotifLogger
from .llm_async_utils import send_llm_async, send_llm_with_validation_async, send_llm_with_images_async
from .library import (
    load,
    length,
)

__all__ = [
    # Visualization functions
    'visualize_scene_motif',
    'visualize_scene_motif_auto_view',
    'generate_all_motif_views',

    # Mesh utilities
    'create_furniture_lookup',
    'assign_mesh_to_object',
    'assign_mesh_to_object_with_normalization',

    # Validation
    'validate_remaining_arrangements',
    'validate_compositional_json',
    'is_sm_exceeds_support_region',
    'inference_validation',

    # Logging
    'MotifLogger',

    # Library functions
    'load',
    'length',
]
