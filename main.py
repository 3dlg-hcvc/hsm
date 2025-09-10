import asyncio
import time
from pathlib import Path
import traceback
from tqdm import tqdm

project_root = Path(__file__).parent

from omegaconf import DictConfig, ListConfig
from argparser import HSMArgumentParser
from hsm_core.scene.processing import (
    setup_scene_generation,
    create_processing_pipeline,
    process_cleanup_stage,
)
from hsm_core.utils import get_logger

async def process_scene(cfg: DictConfig | ListConfig, logger, **kwargs) -> bool:
    """Main execution function for scene generation using function composition."""
    start_time = time.time()

    try:
        logger.info("Starting scene generation pipeline")
        logger.debug(f"Configuration: {cfg}")

        context = await setup_scene_generation(cfg, **kwargs)
        context['start_time'] = start_time
        pipeline = create_processing_pipeline(cfg, context['is_loaded_scene'])

        logger.info(f"Pipeline created with {len(pipeline)} stages")

        with tqdm(total=len(pipeline), desc="Scene Generation", leave=True) as pbar:
            for stage_name, stage_func in pipeline:
                pbar.set_description(stage_name)
                logger.debug(f"Executing stage: {stage_name}")
                context = await stage_func(context, cfg)
                pbar.update(1)
                
        await process_cleanup_stage(context, cfg)
        
        logger.info("Scene generation pipeline completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in process_scene: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main() -> None:
    parser = HSMArgumentParser(project_root)
    args = parser.parse_args()
    cfg = parser.get_config(args)

    logger = get_logger('main')
    print("HSM Scene Generation Started")
    logger.debug(f"Command line arguments: {args}")

    success = asyncio.run(process_scene(cfg, logger, project_root=project_root))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if success:
        print(f"HSM Scene generation completed successfully at {timestamp}!")
    else:
        print(f"HSM Scene generation failed at {timestamp}!")

if __name__ == "__main__":
    main()
