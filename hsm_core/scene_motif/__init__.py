"""
Scene Motif Module

Self contained module that provides scene motif decomposition, generation, and optimization for HSM.
"""

from .core.obj import Obj
from .core.bounding_box import BoundingBox
from .core.arrangement import Arrangement
from .core.hierarchy import MotifHierarchy, HierarchyNode

from .generation import batch_inference, generate_arrangement_code
from .utils.motif_visualize import visualize_scene_motif
from .utils.logger import MotifLogger

from .generation.decomposition import decompose_motif_async, decompose_motif_with_session
from .generation import process_motif_with_visual_validation, build_arrangement_from_json
from .spatial import optimize_sm
from .spatial.spatial_optimizer import optimize
from .spatial.hierarchical_optimizer import optimize_with_hierarchy
from .programs import Program, execute
from .utils.motif_visualize import generate_all_motif_views
from .utils.mesh_utils import create_furniture_lookup, assign_mesh_to_object
from .utils.utils import (
    log_time, calculate_arrangement_half_size,
    extract_objects, resolve_sub_arrangements,
    persist_motif_arrangement
)
from .utils.validation import (
    validate_remaining_arrangements,
    validate_compositional_json,
    inference_validation,
)
from .utils.library import load, length

__all__ = [
    # Main public functions
    'batch_inference',
    'generate_arrangement_code',

    # Core data structures
    'Obj', 'BoundingBox', 'Arrangement',
    'MotifHierarchy', 'HierarchyNode',

    # Public utilities
    'visualize_scene_motif',
    'MotifLogger'
]

__version__ = "1.0.0"
