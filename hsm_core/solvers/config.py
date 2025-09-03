from dataclasses import dataclass
from typing import Dict, Optional
from hsm_core.scene.core.objecttype import ObjectType


@dataclass
class SceneSpatialOptimizerConfig:
    """Configuration for scene level spatial optimization."""
    
    # Core optimization settings
    use_motif_level_optimization: bool = True
    debug_output: bool = True
    
    # Collision detection settings
    collision_tolerance: float = 0.001  # 1cm tolerance for bounding box collision fallback
    
    # Performance optimization settings
    enable_early_exit_optimization: bool = True
    penetration_depth_tolerance: float = 0.003  # 3mm tolerance for early exit
    
    # Cache settings
    enable_collision_manager_cache: bool = True
    enable_mesh_caching: bool = True
    
    # Raycast settings
    raycast_distance: float = 2.0  # Maximum raycast distance in meters
    
    # Iteration settings
    max_collision_iterations: int = 50  # Maximum collision resolution iterations
    
    # Support validation tolerances
    support_tolerance: float = 0.01  # threshold consider object supported
    
    # Movement constraints
    wall_y_movement_factor: float = 0.1  # Factor for Y movement of wall objects
    
    # Adaptive step size settings
    adaptive_step_factor: float = 0.6  # Factor for adaptive step size based on penetration
    step_reduction_factor: float = 0.8  # Factor for reducing step size when no progress
    step_reduction_threshold: float = 0.95  # Threshold for considering no progress
    min_step_size: float = 0.0001  # Minimum step size (0.1mm) - exit loop if step becomes smaller
    
    # Collision resolution movement settings
    vertical_step_factor: float = 0.5  # Factor for vertical step movement
    # max_vertical_adjustment: float = max_motif_translation
    horizontal_step_factor: float = 0.3  # Factor for horizontal step movement  
    # max_horizontal_adjustment: float = max_motif_translation
    
    # Support fixing settings
    support_stability_offset: float = 0.005  # 5mm offset for stability
    support_position_threshold: float = support_tolerance
    
    # Object type specific step sizes (used by _get_step_size method)
    step_sizes: Optional[Dict[ObjectType, float]] = None
    
    def __post_init__(self):
        """Initialize default step sizes if not provided."""
        if self.step_sizes is None:
            self.step_sizes = {
                ObjectType.LARGE: 0.01,      # 1cm - minimal adjustment for large objects
                ObjectType.WALL: 0.008,      # 8mm - very small adjustment for wall objects  
                ObjectType.CEILING: 0.008,   # 8mm - very small adjustment for ceiling objects
                ObjectType.SMALL: 0.005,     # 5mm - tiny adjustment for small objects
            }
    
    def get_step_size(self, obj_type: ObjectType) -> float:
        """Get step size for object type."""
        if self.step_sizes is None:
            # Fallback in case __post_init__ wasn't called
            return 0.01
        return self.step_sizes.get(obj_type, 0.01)
    
    def get_support_tolerance(self, obj_type: ObjectType) -> float:
        """Get support tolerance for object type."""
        return self.support_tolerance
    
    @classmethod
    def create_debug_config(cls) -> 'SceneSpatialOptimizerConfig':
        return cls(debug_output=True)