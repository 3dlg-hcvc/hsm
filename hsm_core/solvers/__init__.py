"""
Modular Spatial Optimization System

This package provides a flexible, configurable spatial optimization system for 3D scenes.
The system uses an advanced mesh-based collision detection engine with room geometry integration.

Key Components:
- UnifiedSceneSpatialOptimizer: The primary, mesh-based spatial optimizer.
- SpatialOptimizerConfig: A simple configuration system for the optimizer.

Usage:
    from hsm_core.solvers import UnifiedSceneSpatialOptimizer, SpatialOptimizerConfig
    
    config = SpatialOptimizerConfig()
    optimizer = UnifiedSceneSpatialOptimizer(scene, config)
    optimized_objects = optimizer.optimize_objects(objects)
"""

# Main optimizer (stays in solvers)
from .unified_optimizer import SceneSpatialOptimizer

# Spatial optimizer (moved to scene_motif.spatial)
from hsm_core.scene_motif.spatial.spatial_optimizer import optimize

# Hierarchical optimizer (moved to scene_motif.spatial)
from hsm_core.scene_motif.spatial.hierarchical_optimizer import optimize_with_hierarchy

# Configuration
from .config import SceneSpatialOptimizerConfig

# Public API
__all__ = [
    # Main optimizer
    'SceneSpatialOptimizer',

    # Spatial optimizer
    'optimize',

    # Hierarchical optimizer
    'optimize_with_hierarchy',

    # Configuration
    'SceneSpatialOptimizerConfig',
]
