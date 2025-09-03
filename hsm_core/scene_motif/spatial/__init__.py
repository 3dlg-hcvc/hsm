"""
Spatial Optimization for scene motifs
"""

from .hierarchical_optimizer import optimize_with_hierarchy
from .spatial_optimizer import optimize

def optimize_sm(arrangement, hierarchy=None, **kwargs):
    """
    Spatial optimization entry point

    Args:
        arrangement: The arrangement to optimize
        hierarchy: Optional MotifHierarchy for hierarchical optimization
        **kwargs: Additional optimization parameters

    Returns:
        Optimized arrangement
    """
    # Check if any objects have meshes before attempting optimization
    objs_with_meshes = [obj for obj in arrangement.objs if hasattr(obj, 'has_mesh') and obj.has_mesh]
    if not objs_with_meshes:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Skipping spatial optimization - no objects have meshes. Objects: {[obj.label for obj in arrangement.objs]}")
        return arrangement  # Return unchanged arrangement

    if hierarchy and hierarchy.root:
        return optimize_with_hierarchy(arrangement, hierarchy=hierarchy, **kwargs)
    else:
        return optimize(arrangement, **kwargs)

__all__ = [
    'optimize_sm'
]
