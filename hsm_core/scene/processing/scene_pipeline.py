import sys
import time
from omegaconf import DictConfig, ListConfig
from matplotlib import pyplot as plt

from hsm_core.retrieval.model.model_manager import ModelManager
from hsm_core.scene.processing.ceiling import process_ceiling_objects
from hsm_core.scene.processing.small import process_small_objects
from hsm_core.scene.processing.large import process_large_objects
from hsm_core.scene.processing.wall import process_wall_objects
from hsm_core.scene.setup.setup import initialize_scene_from_config, perform_room_analysis_and_decomposition
from hsm_core.utils import get_logger

logger = get_logger('scene.pipeline')

async def setup_scene_generation(cfg: DictConfig | ListConfig, **kwargs) -> dict:
    """Setup phase for scene generation."""
    project_root = kwargs.get('project_root')
    if project_root is None:
        raise ValueError("project_root must be provided")

    setup_result = initialize_scene_from_config(
        cfg=cfg, project_root=project_root,
        output_dir_name_override=kwargs.get('output_dir_name_override'),
        output_dir_override=kwargs.get('output_dir_override'),
        timestamp=kwargs.get('timestamp', True)
    )

    vis_output_dir = setup_result.output_dir / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    model = kwargs.get('model')
    if model is None:
        logger = get_logger('scene.pipeline')
        logger.info("Initializing local CLIP model")
        model = await ModelManager.get_clip_model_async()

    return {
        'scene': setup_result.scene,
        'output_dir_override': setup_result.output_dir,
        'room_session': setup_result.room_session,
        'visualizer': setup_result.visualizer,
        'sessions_dir': setup_result.sessions_dir,
        'is_loaded_scene': setup_result.is_loaded_scene,
        'logger': setup_result.logger,
        'vis_output_dir': vis_output_dir,
        'model': model,
        'current_room_plot': None,
        'room_polygon': setup_result.scene.room_polygon,
        'sessions': {},
        'project_root': project_root
    }

async def process_room_analysis(context: dict, cfg: DictConfig) -> dict:
    """Process room analysis and decomposition."""
    logger = get_logger('scene.room_analysis')

    if not context['is_loaded_scene']:
        logger.info(f"Starting room analysis for: {cfg.room.room_description}")
        logger.debug(f"Object types to generate: {cfg.mode.object_types}")
        logger.debug(f"Extra object types: {cfg.mode.extra_types}")

        if all(k in context for k in ['room_session', 'visualizer', 'output_dir_override', 'project_root']):
            logger.debug("Performing room analysis and decomposition")
            context['current_room_plot'], _ = perform_room_analysis_and_decomposition(
                scene=context['scene'],
                room_session=context['room_session'],
                project_root=context['project_root'],
                visualizer=context['visualizer'],
                vis_output_dir=context['vis_output_dir']
            )
            logger.info("Room analysis and decomposition completed")
        else:
            logger.warning("Missing required context keys for room analysis")

    else:
        logger.info("⏭Skipping room analysis (scene already loaded)")

    return context

async def process_floor_support_region_stage(context: dict, cfg: DictConfig) -> dict:
    """Process floor support region stage."""
    if "large" in cfg.mode.object_types:
        required_keys = ['model', 'visualizer', 'output_dir_override', 'sessions_dir', 'room_polygon', 'current_room_plot']
        if all(k in context for k in required_keys):
            context['current_room_plot'], context['sessions']['floor_session'] = await process_large_objects(
                scene=context['scene'], cfg=cfg,
                output_dir_override=context['output_dir_override'],
                room_description=context['scene'].room_description,
                model=context['model'], visualizer=context['visualizer'],
                room_polygon=context['room_polygon'], current_room_plot=context['current_room_plot'],
                sessions_dir=str(context['sessions_dir']), vis_output_dir=context['vis_output_dir']
            )
    else:
        logger.info("Skipping floor support region from config")
    return context

async def process_wall_support_region_stage(context: dict, cfg: DictConfig) -> dict:
    """Process wall support region stage."""
    if "wall" in cfg.mode.object_types:
        required_keys = ['model', 'visualizer', 'output_dir_override', 'sessions_dir', 'current_room_plot']
        if all(k in context for k in required_keys):
            context['sessions']['wall_session'] = await process_wall_objects(
                scene=context['scene'], cfg=cfg,
                output_dir_override=context['output_dir_override'], vis_output_dir=context['vis_output_dir'],
                room_description=context['scene'].room_description, model=context['model'],
                visualizer=context['visualizer'], current_room_plot=context['current_room_plot'],
                sessions_dir=str(context['sessions_dir'])
            )
    else:
        logger.info("Skipping wall support regions from config")
    return context

async def process_ceiling_support_region_stage(context: dict, cfg: DictConfig) -> dict:
    """Process ceiling support region stage."""
    if "ceiling" in cfg.mode.object_types:
        required_keys = ['model', 'visualizer', 'output_dir_override', 'sessions_dir', 'room_polygon', 'current_room_plot']
        if all(k in context for k in required_keys):
            context['sessions']['ceiling_session'] = await process_ceiling_objects(
                scene=context['scene'], cfg=cfg,
                output_dir_override=context['output_dir_override'],
                room_description=context['scene'].room_description, model=context['model'],
                visualizer=context['visualizer'], room_polygon=context['room_polygon'],
                current_room_plot=context['current_room_plot'], sessions_dir=str(context['sessions_dir']),
                vis_output_dir=context['vis_output_dir']
            )
    else:
        logger.info("Skipping ceiling support region from config")
    return context

async def process_furniture_support_regions_stage(context: dict, cfg: DictConfig) -> dict:
    """Process furniture support region stage."""
    if "small" in cfg.mode.object_types:
        if context.get('model') and context.get('output_dir_override'):
            context['sessions']['small_session'] = await process_small_objects(
                scene=context['scene'], cfg=cfg,
                output_dir_override=context['output_dir_override'], model=context['model'],
                vis_output_dir=context['vis_output_dir'], sessions_dir=str(context['sessions_dir'])
            )
            context['scene'].save(context['output_dir_override'])
    else:
        logger.info("Skipping furniture support regions from config")
        if context.get('output_dir_override'):
            context['scene'].save(context['output_dir_override'])
    return context

async def process_cleanup_stage(context: dict, cfg: DictConfig|ListConfig) -> dict:
    """Cleanup stage in the processing pipeline."""
    end_time = time.time()
    start_time = context.get('start_time', time.time())
    minutes, seconds = divmod(end_time - start_time, 60)
    print("Scene saved to ", context['output_dir_override'])
    logger.info(f"\nTime taken: {minutes:.0f}m {seconds:.0f}s")

    # logger.info("Saving all VLM sessions")
    # if context.get('room_session') and "large" in cfg.mode.object_types:
    #     context['room_session'].save_session()
    # for session_name in ['floor_session', 'wall_session', 'ceiling_session', 'small_session']:
    #     if session := context.get('sessions', {}).get(session_name):
    #         session.save_session()

    ModelManager.clear_cache()
    plt.close("all")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if output_dir := context.get('output_dir_override'):
        logger.info(f"Log saved to: {output_dir / 'scene.log'}")

    return context

def create_processing_pipeline(cfg: DictConfig | ListConfig, is_loaded_scene: bool) -> list:
    stages = []

    if not is_loaded_scene:
        stages.append(("Requirements Decomposition", process_room_analysis))

    stage_map = {
        "large": ("Floor Support Region", process_floor_support_region_stage),
        "wall": ("Wall Support Regions", process_wall_support_region_stage),
        "ceiling": ("Ceiling Support Region", process_ceiling_support_region_stage),
        "small": ("Furniture Support Regions", process_furniture_support_regions_stage)
    }

    for obj_type, (stage_name, stage_func) in stage_map.items():
        if obj_type in cfg.mode.object_types:
            stages.append((stage_name, stage_func))
            
    return stages