# Scene Motif Module

Scene motif generation, processing, and spatial optimization for HSM.
Some code adapted from SceneMotifCoder (SMC).

## File Structure

```
scene_motif/
├── core/
│   ├── obj.py           # Object representation
│   ├── arrangement.py   # Object collections
|   |── bounding_box.py  # Bounding box
│   ├── hierarchy.py     # Hierarchical motif structure
├── generation/     # scene motif generation
│   ├── llm/            # LLM utilities and generators
│   └── processing/     # Main processing logic
│   ├── decomposition.py # Motif decomposition
├── spatial/        # Spatial optimization
│   ├── spatial_optimizer.py      # Core optimization
│   ├── hierarchical_optimizer.py # Hierarchical optimization
├── programs/       # Program interpreters and validators
└── utils/          # Shared utilities, visualization, and library management
```

## Data

The motif library data is stored in the root `data/motif_library/` directory:

```
data/
├── motif_library/
    └── meta_programs/ # Meta-programs for motif types
```

## Run Tests
```bash
python -m hsm_core.scene_motif.generation.processing.batch_inference 
```