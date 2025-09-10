"""
Small Object Processing Module
"""

from pathlib import Path
from omegaconf import DictConfig

from hsm_core.scene.core.manager import Scene
from hsm_core.config import PROMPT_DIR, PROJECT_ROOT
from hsm_core.retrieval.model.model_manager import ModelManager
from hsm_core.vlm.gpt import Session
from hsm_core.utils import get_logger

logger = get_logger('scene.small')

async def process_small_objects(
    scene: Scene,
    cfg: DictConfig,
    output_dir_override: Path,
    model: ModelManager,
    vis_output_dir: Path,
    sessions_dir: Path,
) -> Session:
    """
    Process small objects for the scene.
    
    Args:
        scene: Scene object containing room and object data
        cfg: Configuration object
        output_dir_override: Output directory path
        model: ModelManager instance for CLIP model
        vis_output_dir: Output directory path for visualizations
        sessions_dir: Output directory path for VLM sessions
    Returns:
        Session object used for small object processing
    """
    logger.info("Processing Small Objects...")

    if "small" not in cfg.mode.object_types:
        logger.info("Small objects not in processing types, skipping...")
        dummy_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))
        dummy_session.output_dir = str(sessions_dir)
        return dummy_session

    try:
        small_session = await scene.populate_small_objects(cfg, str(output_dir_override), vis_output_dir, model)

        if small_session is None:
            # Create a session for consistency with other object types
            small_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))
            small_session.output_dir = str(sessions_dir)

        logger.info("Small object processing completed")
        return small_session

    except Exception as e:
        logger.error(f"Error during small object processing: {e}")
        import traceback
        traceback.print_exc()
        
        dummy_session = Session(str(PROMPT_DIR / "scene_prompts_small.yaml"))
        dummy_session.output_dir = str(sessions_dir)
        return dummy_session