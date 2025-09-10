# Re-export classes from specifications module to maintain backward compatibility
from hsm_core.scene.specifications.scene_spec import SceneSpec
from hsm_core.scene.specifications.object_spec import ObjectSpec
from hsm_core.utils import get_logger
import pprint

# Explicit re-exports for backward compatibility
__all__ = ['SceneSpec', 'ObjectSpec']

logger = get_logger('scene.core.spec')

        
if __name__ == "__main__":
    test_response = """{
  "objects": [
    {
      "id": 1,
      "name": "bookcase",
      "description": "large wooden shelf",
      "dimensions": [
        1,
        2,
        0.3
      ],
      "amount": 1
    },
    {
      "id": 2,
      "name": "test",
      "description": "test",
      "dimensions": [
        1,
        2,
        0.3
      ],
      "amount": 1
    }
  ],
  "wall_objects": [],
  "ceiling_objects": [],
  "small_objects": [
    {
      "id": 2,
      "name": "book",
      "description": "standard hardcover book",
      "dimensions": [
        0.15,
        0.02,
        0.22
      ],
      "amount": 8,
      "parent_object": 2
    },
    {
      "id": 3,
      "name": "lamp",
      "description": "small table lamp",
      "dimensions": [
        0.15,
        0.3,
        0.15
      ],
      "amount": 1,
      "parent_object": 2
    },
    {
      "id": 4,
      "name": "vase",
      "description": "small decorative vase",
      "dimensions": [
        0.1,
        0.2,
        0.1
      ],
      "amount": 1,
      "parent_object": 2
    },
    {
      "id": 5,
      "name": "plant",
      "description": "small potted plant",
      "dimensions": [
        0.2,
        0.3,
        0.2
      ],
      "amount": 1,
      "parent_object": 2
    },
    {
      "id": 6,
      "name": "bottle",
      "description": "standard glass bottle",
      "dimensions": [
        0.08,
        0.25,
        0.08
      ],
      "amount": 1,
      "parent_object": 2
    }
  ]
}"""
    
    scene_spec = SceneSpec.from_json(str(test_response), required=True)
    logger.debug(f"Scene spec: {scene_spec}")
    logger.debug(f"Layered small objects: {scene_spec.layered_small_objects}")