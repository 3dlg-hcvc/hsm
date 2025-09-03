"""
Small Object Processing Module
"""

from pathlib import Path
from omegaconf import DictConfig

from hsm_core.scene.manager import Scene
from hsm_core.config import PROMPT_DIR, PROJECT_ROOT
from hsm_core.retrieval.model.model_manager import ModelManager
from hsm_core.vlm.gpt import Session

async def process_small_objects(
    scene: Scene,
    cfg: DictConfig,
    output_dir_override: Path,
    model: ModelManager
) -> Session:
    """
    Process small objects for the scene.
    
    Args:
        scene: Scene object containing room and object data
        cfg: Configuration object
        output_dir_override: Output directory path
        model: ModelManager instance for CLIP model
        
    Returns:
        Session object used for small object processing
    """
    print("\nProcessing Small Objects...")
    
    if "small" not in cfg.mode.object_types:
        print("Small objects not in processing types, skipping...")
        # Create a dummy session for consistency
        dummy_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))
        dummy_session.output_dir = str(output_dir_override / ".sessions")
        return dummy_session
    
    try:
        small_session = await scene.populate_small_objects(cfg, str(output_dir_override), model)
        
        if small_session is None:
            # Create a session for consistency with other object types
            small_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))
            small_session.output_dir = str(output_dir_override / ".sessions")
        
        print("Small object processing completed")
        return small_session
        
    except Exception as e:
        print(f"Error during small object processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a dummy session for consistency
        dummy_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))
        dummy_session.output_dir = str(output_dir_override / ".sessions")
        return dummy_session