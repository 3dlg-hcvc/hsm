import asyncio
import gc
import os
import sys
import time
import traceback
from typing import Any
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from omegaconf import DictConfig
from shapely.geometry import Polygon

from hsm_core.scene.setup import initialize_scene_from_config, perform_room_analysis_and_decomposition
from hsm_core.scene.large import process_large_objects
from hsm_core.scene.wall import process_wall_objects
from hsm_core.scene.ceiling import process_ceiling_objects
from hsm_core.scene.small import process_small_objects

from hsm_core.retrieval.model.model_manager import ModelManager
from argparser import HSMArgumentParser

async def process_scene(
    cfg: DictConfig,
    output_dir_override: Path | None = None,
    output_dir_name_override: str | None = None,
    timestamp: bool = True,
    model: Any | None = None
) -> bool:
    """
    Main execution function for scene generation.

    Args:
        cfg: OmegaConf configuration object from scene_config.yaml
        output_dir_override: Path to the output directory
        output_dir_name_override: Name of the output directory
        timestamp: Whether to include a timestamp in the output directory name (default: True)
        model: Optional pre-initialized ModelManager instance.
    """
    start_time = time.time()
    logger = None

    try:
        # Initialize scene from configuration
        setup_result = initialize_scene_from_config(
            cfg=cfg,
            project_root=project_root,
            output_dir_name_override=output_dir_name_override,
            output_dir_override=output_dir_override,
            timestamp=timestamp
        )
        
        scene = setup_result.scene
        output_dir_override = setup_result.output_dir
        room_session = setup_result.room_session
        visualizer = setup_result.visualizer
        sessions_dir = setup_result.sessions_dir
        is_loaded_scene = setup_result.is_loaded_scene
        logger = setup_result.logger
        updated_plot: Figure | None = None
        room_polygon: Polygon = scene.room_polygon

        if model is None:
            print("Initializing local CLIP model...")
            model = await ModelManager.get_clip_model_async()
        
        processing_stages = []
        if not is_loaded_scene:
            processing_stages.append("Room Analysis")
        if "large" in cfg.mode.object_types:
            processing_stages.append("Large Objects")
        if "wall" in cfg.mode.object_types:
            processing_stages.append("Wall Objects")
        if "ceiling" in cfg.mode.object_types:
            processing_stages.append("Ceiling Objects")
        if "small" in cfg.mode.object_types:
            processing_stages.append("Small Objects")
        
        is_batch = os.environ.get("HSM_BATCH_MODE") == "1"
        with tqdm(total=len(processing_stages), desc="Scene Generation", leave=True, disable=is_batch) as pbar:
            print("config", cfg)
            
            # Perform room analysis and decomposition for new scenes
            if not is_loaded_scene:
                pbar.set_description("Room Analysis")
                print(f"Start generation for room: {cfg.room.room_description}")
                print(f"Generating object types: {cfg.mode.object_types}")
                print(f"Extra object types: {cfg.mode.extra_types}\n")
                print("="*50)

                if room_session and visualizer and output_dir_override:
                    updated_plot, room_details = perform_room_analysis_and_decomposition(
                        scene=scene,
                        room_session=room_session,
                        project_root=project_root,
                        visualizer=visualizer,
                        output_dir=output_dir_override
                    )
                pbar.update(1)
            else:
                print("Skipping room analysis and decomposition from config")
                if visualizer and output_dir_override:
                    initial_plot_path = output_dir_override / "initial_plot.png"
                    updated_plot, _ = visualizer.visualize(
                        output_path=str(initial_plot_path), add_grid_markers=True
                    )

            # Process Large Objects if enabled
            if "large" in cfg.mode.object_types:
                pbar.set_description("Floor Support Region")
                if model and visualizer and output_dir_override and sessions_dir and room_polygon and updated_plot:
                    plot_after_large, large_session = await process_large_objects(
                        scene=scene,
                        cfg=cfg,
                        output_dir_override=output_dir_override,
                        room_description=scene.room_description,
                        model=model,
                        visualizer=visualizer,
                        room_polygon=room_polygon,
                        initial_plot=updated_plot,
                        project_root=project_root,
                        sessions_dir=str(sessions_dir)
                    )
                    updated_plot = plot_after_large
                pbar.update(1)
            else: 
                print("Skipping large objects from config")

            # Process Wall Objects if enabled
            if "wall" in cfg.mode.object_types:
                pbar.set_description("Wall Support Regions")
                if model and visualizer and output_dir_override and sessions_dir and updated_plot:
                    wall_session = await process_wall_objects(
                        scene=scene,
                        cfg=cfg,
                        output_dir_override=output_dir_override,
                        room_description=scene.room_description,
                        model=model,
                        visualizer=visualizer,
                        updated_plot=updated_plot,
                        project_root=project_root,
                        sessions_dir=str(sessions_dir)
                    )
                pbar.update(1)
            else: 
                print("Skipping wall objects from config")
            
            # Process Ceiling Objects if enabled
            if "ceiling" in cfg.mode.object_types:
                pbar.set_description("Ceiling Support Region")
                if model and visualizer and output_dir_override and sessions_dir and room_polygon and updated_plot:
                    ceiling_session = await process_ceiling_objects(
                        scene=scene,
                        cfg=cfg,
                        output_dir_override=output_dir_override,
                        room_description=scene.room_description,
                        model=model,
                        visualizer=visualizer,
                        room_polygon=room_polygon,
                        updated_plot=updated_plot,
                        project_root=project_root,
                        sessions_dir=str(sessions_dir)
                    )
                pbar.update(1)
            else: 
                print("Skipping ceiling objects from config")
            
            # Process Small Objects if enabled
            if "small" in cfg.mode.object_types:
                pbar.set_description("Furniture Support Regions")
                if model and output_dir_override:
                    small_session = await process_small_objects(
                        scene=scene,
                        cfg=cfg,
                        output_dir_override=output_dir_override,
                        model=model
                    )
                    if output_dir_override:
                        # Only save scene state at the very end
                        scene.save(output_dir_override, "full", save_scene_state=True)
                pbar.update(1)
            else: 
                print("Skipping small objects from config")
                # If no small objects, save scene state here
                if output_dir_override:
                    scene.save(output_dir_override, save_scene_state=True)
                
        return True

    except Exception as e:
        print(f"Error in process_scene: {str(e)}")
        traceback.print_exc()
        raise
    finally:
        print("\nSaving all active gpt sessions before cleanup")
        if 'room_session' in locals() and room_session and "large" in cfg.mode.object_types:
            room_session.save_session()
        if 'large_session' in locals() and "large" in cfg.mode.object_types:
            large_session.save_session()
        if 'wall_session' in locals() and "wall" in cfg.mode.object_types:
            wall_session.save_session()
        if 'ceiling_session' in locals() and "ceiling" in cfg.mode.object_types:
            ceiling_session.save_session()
        if 'small_session' in locals() and "small" in cfg.mode.object_types:
            small_session.save_session()
            
        try:
            ModelManager.clear_cache()
        except Exception as e:
            print(f"Error releasing model: {e}")

        plt.close("all")
        gc.collect()

        end_time = time.time()
        minutes, seconds = divmod(end_time - start_time, 60)
        print(f"\nTime taken to create the 3D scene: {minutes:.0f}m {seconds:.0f}s")

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if logger:
            logger.close()
        if output_dir_override:
            print(f"Log saved to: {output_dir_override / 'scene_log.log'}")

if __name__ == "__main__":
    parser = HSMArgumentParser(project_root)
    args = parser.parse_args()
    cfg = parser.get_config(args)
    
    cfg.room.room_description = args.desc # use description from args

    # Only force floorplan generation if no vertices are specified in config
    if cfg.room.vertices is None:
        print("No vertices specified in config - will generate floorplan using VLM")
    else:
        print(f"Using vertices from config: {cfg.room.vertices}")
    asyncio.run(process_scene(cfg))
