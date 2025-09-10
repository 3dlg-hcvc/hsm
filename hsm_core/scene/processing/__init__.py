"""
Scene Processing Package

Provides scene generation and processing functionality.
"""

from .scene_pipeline import (
    setup_scene_generation,
    process_room_analysis,
    process_floor_support_region_stage,
    process_wall_support_region_stage,
    process_ceiling_support_region_stage,
    process_furniture_support_regions_stage,
    process_cleanup_stage,
    create_processing_pipeline,
    process_cleanup_stage,
)

__all__ = [
    'setup_scene_generation',
    'process_room_analysis',
    'process_floor_support_region_stage',
    'process_wall_support_region_stage',
    'process_ceiling_support_region_stage',
    'process_furniture_support_regions_stage',
    'process_cleanup_stage',
    'create_processing_pipeline',
    'process_cleanup_stage',
]
