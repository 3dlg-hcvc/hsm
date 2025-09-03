# HSM Core Solvers Module

This module provides a modular, configurable spatial optimization system for 3D scene generation. The spatial optimizer resolves collisions, ensures proper object support, and validates room boundaries while preserving the artistic intent of the original scene layout.

## Architecture Overview

The spatial optimizer has been refactored from a monolithic design into a clean, modular architecture:

```
hsm_core/solvers/
├── __init__.py          # Public API exports
├── config.py            # Configuration system
├── validators.py        # Validation logic (collision, support, boundary)
├── resolvers.py         # Resolution logic (fixing issues)
├── optimizer.py         # Main orchestrator class
└── README.md           # This file
```

## Key Components

### 1. Configuration System (`config.py`)

**`SpatialOptimizerConfig`** - Main configuration class with sensible defaults:
- **Processing Control**: Configurable object type processing order
- **Validation Settings**: Enable/disable collision, support, and boundary validation
- **Adaptive Parameters**: Object-type-specific step sizes and tolerances
- **Optimization Strategies**: Collision-first vs support-first approaches
- **Performance Tuning**: Iteration limits, convergence thresholds

**`ValidationSettings`** - Per-validation-type configuration:
- Collision tolerance and detection methods
- Support validation parameters
- Boundary checking settings

**`ObjectTypeSettings`** - Per-object-type configuration:
- Individual validation settings for LARGE, WALL, CEILING, SMALL objects
- Custom step sizes and movement constraints
- Type-specific optimization parameters

### 2. Validation System (`validators.py`)

**`BaseValidator`** - Common validation functionality
**`CollisionValidator`** - Detects object-object collisions using:
- Mesh-based collision detection (when trimesh available)
- Bounding box fallback
- Surface-aware collision detection for small objects

**`SupportValidator`** - Ensures objects are properly supported:
- Raycast-based support detection
- Floor and surface support validation
- Parent-child relationship validation

**`BoundaryValidator`** - Validates room boundary constraints:
- Room polygon boundary checking
- Wall attachment validation
- Coordinate system consistency

### 3. Resolution System (`resolvers.py`)

**`BaseResolver`** - Common resolution functionality
**`CollisionResolver`** - Fixes collision issues:
- Direction-based collision avoidance
- Object-type-specific step sizes
- Simple but effective collision resolution

**`SupportResolver`** - Fixes support issues:
- Height adjustment for proper support
- Floor and object surface support
- Support height calculation

**`BoundaryResolver`** - Fixes boundary violations:
- Room boundary constraint enforcement
- Position clamping within room bounds
- Boundary tolerance handling

### 4. Main Optimizer (`optimizer.py`)

**`SceneSpatialOptimizer`** - Main orchestration class:
- **Dual Processing Modes**: Individual object vs motif-level optimization
- **Configurable Pipeline**: Collision-first or support-first optimization
- **Statistics Tracking**: Comprehensive optimization metrics
- **Error Handling**: Robust fallback mechanisms

## Usage Examples

### Basic Usage

```python
from hsm_core.solvers import SceneSpatialOptimizer, SpatialOptimizerConfig

# Create configuration
config = SpatialOptimizerConfig(
    enable_collision_detection=True,
    enable_support_validation=True,
    debug_output=True
)

# Create optimizer
optimizer = SceneSpatialOptimizer(scene, config)

# Optimize objects
optimized_objects = optimizer.optimize_objects(scene_objects)
```

### Advanced Configuration

```python
# Custom processing order
config = SpatialOptimizerConfig()
config.set_processing_order([ObjectType.SMALL, ObjectType.LARGE, ObjectType.WALL])

# Per-object-type settings
config.configure_validation(ObjectType.SMALL, "collision", tolerance=0.005)
config.configure_validation(ObjectType.LARGE, "support", tolerance=0.02)

# Enable/disable object types
config.enable_object_type(ObjectType.CEILING, False)

# Optimization strategy
config.optimization_strategy = OptimizationStrategy.COLLISION_FIRST
```

### Motif-Level Optimization

```python
# Enable motif-level optimization for better structure preservation
config = SpatialOptimizerConfig(
    use_motif_level_optimization=True,
    motif_step_size=0.01,
    preserve_motif_internal_structure=True
)

optimizer = SceneSpatialOptimizer(scene, config)
optimized_objects = optimizer.optimize_objects(scene_objects)
```

### Pre-built Configurations

```python
# Minimal optimization (only critical issues)
config = SpatialOptimizerConfig.create_minimal_config()

# Comprehensive optimization (all validations)
config = SpatialOptimizerConfig.create_comprehensive_config()

# Collision-only optimization
config = SpatialOptimizerConfig.create_collision_only_config()
```

## Key Features

### 🎯 **Flexible Processing Control**
- Configurable object type processing order
- Enable/disable specific object types
- Per-object-type validation settings

### 🔧 **Modular Validation & Resolution**
- Separate validation and resolution logic
- Pluggable validator and resolver components
- Type-safe configuration system

### 📊 **Comprehensive Statistics**
- Before/after validation metrics
- Processing time tracking
- Detailed debug logging

### 🎨 **Artistic Intent Preservation**
- Motif-level optimization maintains object relationships
- Configurable solver position preservation
- Minimal optimization modes for well-placed objects

### ⚡ **Performance Optimization**
- Mesh caching for repeated operations
- Progressive step size reduction
- Early termination on convergence

### 🛡️ **Robust Error Handling**
- Graceful fallbacks for mesh loading failures
- Exception isolation between objects
- Comprehensive logging for debugging

## Integration with Scene Generation

The spatial optimizer integrates seamlessly with the scene generation pipeline:

1. **Scene Creation**: Objects are initially placed by motif generators and solvers
2. **Spatial Optimization**: The optimizer refines positions to resolve issues
3. **Final Validation**: Comprehensive validation ensures scene quality

```python
# In scene_3d.py
from hsm_core.solvers import SceneSpatialOptimizer, SpatialOptimizerConfig

# Configure optimization
optimizer_config = SpatialOptimizerConfig()

# Create optimizer and apply optimization
optimizer = SceneSpatialOptimizer(scene_manager, optimizer_config, output_dir)
optimizer.set_preprocessed_meshes(preprocessed_meshes)
optimized_objects = optimizer.optimize_objects(all_objects)
```

## Backward Compatibility

The new modular system is fully backward compatible with existing code:

```python
# This still works - automatically uses the new modular optimizer
from hsm_core.solvers import SceneSpatialOptimizer, SpatialOptimizerConfig

# Legacy import also works
from hsm_core.solvers.scene_spatial_optimizer import SceneSpatialOptimizer

# The API is identical - no code changes required
optimizer = SceneSpatialOptimizer(scene, config)
results = optimizer.optimize_objects(objects)
```

## Migration Guide

### For New Code
Use the modular system with enhanced configuration:

```python
from hsm_core.solvers import SceneSpatialOptimizer, SpatialOptimizerConfig
from hsm_core.solvers import OptimizationStrategy, ObjectType

config = SpatialOptimizerConfig(
    strategy=OptimizationStrategy.COLLISION_FIRST,
    use_motif_level_optimization=True,
    debug_output=True
)

# Configure per-object-type settings
config.configure_validation(ObjectType.SMALL, "collision", tolerance=0.005)
config.enable_object_type(ObjectType.CEILING, False)

optimizer = SceneSpatialOptimizer(scene, config)
```

### For Existing Code
No changes required - the system automatically uses the new modular optimizer while maintaining the same API.

## Performance Improvements

The new modular system provides several performance benefits:

- **Selective Processing**: Only enabled object types are processed
- **Early Termination**: Skip optimization for well-placed objects
- **Configurable Iterations**: Adjust iteration limits per object type
- **Minimal Mode**: Conservative optimization for small objects
- **Statistics Tracking**: Monitor performance and optimization effectiveness

## Debugging & Troubleshooting

Enable debug output for detailed optimization information:

```python
config = SpatialOptimizerConfig(debug_output=True)
optimizer = SceneSpatialOptimizer(scene, config)
optimized_objects = optimizer.optimize_objects(scene_objects)

# Check optimization statistics
print(f"Statistics: {optimizer.stats}")
```

The optimizer provides comprehensive statistics including:
- Objects processed and positions adjusted
- Collisions and support issues resolved
- Processing time and validation metrics
- Before/after comparison data

## Future Enhancements

The modular architecture enables easy extension:
- Additional validation types (e.g., aesthetic constraints)
- New resolution strategies (e.g., physics-based simulation)
- Custom optimization algorithms
- Integration with external physics engines 