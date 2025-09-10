import json

from hsm_core.vlm.gpt import Session, extract_json
from hsm_core.scene.core.motif import SceneMotif
from hsm_core.scene.core.objects import SceneObject
from hsm_core.config import PROMPT_DIR
from hsm_core.utils import get_logger

logger = get_logger('scene.utils.anchor')


def find_anchor_object(scene_motif: SceneMotif, object_names: list[str] = []) -> tuple[list[SceneObject], list[str]]:
    """
    Find anchor objects in the scene by querying GPT to identify key objects.

    Args:
        scene_motif (SceneMotif): The scene motif containing scene objects

    Returns:
        tuple: (list[SceneObject], list[str]) - List of scene objects and their names identified as anchors
        
    Note:
        Uses GPT to identify important objects in the scene based on object names.
        Returns [], [] if no valid objects are found.
    """
    # Initialize GPT session with small objects prompt
    small_obj_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))
    
    if not object_names:
        _, all_object_names = scene_motif.get_objects_by_names()
    else:
        all_object_names = object_names
    
    response = small_obj_session.send("choose_objects", {
        "OBJECT_LIST": all_object_names
    }, is_json=True, verbose=True)
    
    # Extract JSON from response
    json_data = json.loads(extract_json(response))    
    object_names = json_data.get("objects", [])

    if isinstance(json_data.get("objects"), list):
        if not isinstance(object_names, list) or not object_names:
            return [], []
    
    return scene_motif.get_objects_by_names(object_names)
