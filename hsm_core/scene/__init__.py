"""
HSM Core Scene Module
"""

from .setup.setup import initialize_scene_from_config, perform_room_analysis_and_decomposition
from .processing.large import process_large_objects
from .processing.wall import process_wall_objects
from .processing.ceiling import process_ceiling_objects
from .processing.small import process_small_objects
from .processing.scene_pipeline import setup_scene_generation, create_processing_pipeline, process_cleanup_stage
from .core.manager import Scene
from .core.motif import SceneMotif
from .core.objects import SceneObject
from .core.objecttype import ObjectType
from .core.spec import ObjectSpec, SceneSpec
from .geometry.cutout import Cutout

__all__ = [
    'initialize_scene_from_config',
    'perform_room_analysis_and_decomposition',
    'process_large_objects',
    'process_wall_objects',
    'process_ceiling_objects',
    'process_small_objects',
    'Scene',
    'SceneMotif',
    'SceneObject',
    'ObjectType',
    'SceneSpec',
    'ObjectSpec',
    'Cutout',
    'setup_scene_generation',
    'create_processing_pipeline',
    'process_cleanup_stage'
]