"""
Batch inference for scene motifs
"""

import asyncio
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
from hsm_core.utils import get_logger

from ...utils.utils import log_time
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.scene.specifications.object_spec import ObjectSpec
from hsm_core.config import HSSD_PATH, PROJECT_ROOT

logger = get_logger('scene_motif.generation.processing.batch_inference')

class SceneMotif:
    """Minimal SceneMotif implementation for testing"""

    def __init__(self, id: str, description: str, object_specs: list, extents: tuple):
        self.id = id
        self.description = description
        self.object_specs = object_specs
        self.extents = extents
        self.position = (0, 0, 0)
        self.rotation = 0
        self.height_limit = 0.0  # Add this for compatibility

    def __repr__(self):
        return f"SceneMotif(id='{self.id}', description='{self.description}')"


async def batch_inference(
    scene_motif_specs: List["SceneMotif"],
    output_dir: Path,
    room_type: str,
    model=None,
    save_prefix: str = "",
    optimize: bool = True,
    mesh_overrides: Optional[Dict[str, str]] = None,
    object_type: ObjectType = ObjectType.UNDEFINED,
    log_to_terminal: bool = False,
    support_surface_constraints: Optional[Dict[str, Dict]] = None,
    skip_visual_validation: bool = False,
    force_make_tight: bool = False
) -> List["SceneMotif"]:
    """Process multiple scene motifs in batch"""
    total_start = time.time()
    logger.info("Starting batch inference...")

    if model is None:
        logger.info("Initializing CLIP model...")
        from hsm_core.retrieval.model.model_manager import ModelManager
        model = await ModelManager.get_clip_model_async()

    from .motif_processing import process_single_object_motifs
    multi_object_motifs, processed_motifs, furniture_mapping, all_furniture_specs = await process_single_object_motifs(
        scene_motif_specs, output_dir, model, save_prefix, mesh_overrides,
        object_type, log_to_terminal, support_surface_constraints
    )

    logger.info("Motif processing results:")
    logger.info(f"  - Input motifs: {len(scene_motif_specs)}")
    logger.info(f"  - Multi-object motifs: {len(multi_object_motifs)}")
    logger.info(f"  - Processed single-object motifs: {len(processed_motifs)}")
    logger.info(f"  - All furniture objects: {len(all_furniture_specs)}")

    if processed_motifs:
        logger.info("Successfully processed single-object motifs:")
        for motif in processed_motifs:
            logger.info(f"  - {motif.id}: {motif.description}")

    if not multi_object_motifs:
        logger.info("No multi-object motifs to process")
        return processed_motifs

    decomposition_start = time.time()
    logger.info("Starting decomposition and retrieval...")

    # Create semaphore to limit concurrent VLM calls
    semaphore = asyncio.Semaphore(5)

    from ..decomposition import decompose_motif_async
    async def first_decompose_with_semaphore(motif):
        async with semaphore:
            return await decompose_motif_async(motif, max_attempts=1)

    first_decompose_task = asyncio.gather(
        *[first_decompose_with_semaphore(motif) for motif in multi_object_motifs],
        return_exceptions=True
    )

    if all_furniture_specs:
        from hsm_core.retrieval.core.adaptive_retrieval import retrieve_adaptive
        retrieval_task = asyncio.create_task(retrieve_adaptive(
            objs=all_furniture_specs, same_per_label=True, avoid_used=False, randomize=False,
            use_top_k=5, model=model, object_type=object_type,
            max_height=scene_motif_specs[0].height_limit if scene_motif_specs else 0.0,
            support_surface_constraints=support_surface_constraints or {},
            hssd_dir_path=HSSD_PATH
        ))
    else:
        retrieval_task = asyncio.create_task(asyncio.sleep(0))

    decomposition_results, _ = await asyncio.gather(first_decompose_task, retrieval_task)
    # log_time(decomposition_start, f"first decomposition + retrieval")

    # Process motifs with visual feedback
    processing_start = time.time()
    logger.info(f"Starting processing with visual feedback for {len(multi_object_motifs)} motifs...")

    async def process_with_semaphore(motif_data):
        async with semaphore:
            motif, first_decompose_result = motif_data
            from .motif_processing import process_motif_with_visual_validation
            return await process_motif_with_visual_validation(
                motif, furniture_mapping, output_dir, save_prefix, optimize,
                force_make_tight, skip_visual_validation, log_to_terminal, support_surface_constraints,
                first_decompose_result=first_decompose_result, max_attempts=3,
            )

    process_tasks = [
        process_with_semaphore((motif, decomposition_results[i]))
        for i, motif in enumerate(multi_object_motifs)
    ]

    process_results = []
    for task in asyncio.as_completed(process_tasks):
        try:
            result = await task
            process_results.append(result)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            process_results.append(e)

    log_time(processing_start, f"Processing with visual feedback for {len(multi_object_motifs)}  scene motifs")

    for result in process_results:
        if isinstance(result, Exception):
            logger.error(f"Processing failed: {result}")
        elif result is not None and hasattr(result, 'id') and hasattr(result, 'object_specs'):
            processed_motifs.append(result)

    logger.info("="*50)
    logger.info("Batch inference complete!")
    log_time(total_start, f"Total batch inference time for {len(scene_motif_specs)} scene motifs")
    logger.info("="*50)
    return processed_motifs 

def main():
    large_test_data = [
        {
            'id': 'sofa_tables_01', 'area_name': 'Sofa Area',
            'composition': {
                'description': 'a sofa in front of a coffee table with two side tables on each side',
                'furniture': [
                    ObjectSpec(id=1, name='sofa', description='large sofa', dimensions=[2.2, 0.9, 1.0], amount=1),
                    ObjectSpec(id=2, name='side_table', description='small side table', dimensions=[0.5, 0.6, 0.5], amount=2),
                    ObjectSpec(id=3, name='coffee_table', description='standard coffee table', dimensions=[1.2, 0.4, 0.6], amount=1)
                ],
                'total_footprint': [3.2, 0.9, 2.0], 'clearance': 0.5
            },
            'rationale': 'Creates a balanced living room arrangement'
        },
        {
            'id': 'dining_set_01', 'area_name': 'Dining Area',
            'composition': {
                'description': 'a dining table surrounded by 4 chairs',
                'furniture': [
                    ObjectSpec(id=1, name='dining_table', description='rectangular dining table', dimensions=[1.8, 0.75, 0.9], amount=1),
                    ObjectSpec(id=2, name='chair', description='dining chair', dimensions=[0.5, 0.8, 0.5], amount=4)
                ],
                'total_footprint': [2.8, 0.8, 1.9], 'clearance': 0.5
            },
            'rationale': 'Creates a functional dining arrangement'
        },
        {
            'id': 'bedroom_set_01', 'area_name': 'Bedroom',
            'composition': {
                'description': 'a bed with nightstands on each side',
                'furniture': [
                    ObjectSpec(id=1, name='bed', description='double bed', dimensions=[1.4, 0.6, 2.0], amount=1),
                    ObjectSpec(id=2, name='nightstand', description='bedside table', dimensions=[0.4, 0.6, 0.4], amount=2)
                ],
                'total_footprint': [2.8, 0.6, 1.4], 'clearance': 0.5
            },
            'rationale': 'Creates a bedroom arrangement with symmetrical nightstands'
        }
    ]
    small_test_data = [
    # {
        #     'id': 'stack_of_books_01', 'area_name': 'stack of books',
        #     'composition': {
        #         'description': 'a stack of 4 books',
        #         'furniture': [
        #             ObjectSpec(id=1, name='book', description='a book', dimensions=[0.3, 0.05, 0.2], amount=4)
        #         ],
        #         'total_footprint': [0.2, 0., 0.05], 'clearance': 0.5
        #     },
        #     'rationale': 'Creates a stack of 4 books'
        # },
        {
        'id': 'place_setting_01',
        'area_name': 'Place Setting',
        'composition': {
            'description': 'a place setting, a plate with a knife and a fork on each side, in front of a cup',
            'furniture': [
                ObjectSpec(id=1, name='plate', description='a plate', dimensions=[0.27, 0.025, 0.27], amount=1), 
                ObjectSpec(id=2, name='knife', description='a knife', dimensions=[0.02, 0.005, 0.23], amount=1), 
                ObjectSpec(id=3, name='fork', description='a fork', dimensions=[0.02, 0.005, 0.20], amount=1),   
                ObjectSpec(id=4, name='cup', description='a cup', dimensions=[0.08, 0.1, 0.08], amount=1),       
            ],
            'total_footprint': [0.45, 0.3, 0.3], 
            'clearance': 0.5
            }
        }
    ]
    skip_visual_validation = True
    force_make_tight = False
    
    # force logging level to INFO
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_large_motifs = [
        SceneMotif(
            id=item['id'], description=item['composition']['description'],
            object_specs=item['composition']['furniture'],
            extents=tuple(item['composition']['total_footprint'])
        ) for item in large_test_data
    ]
    
    test_small_motifs = [
        SceneMotif(
            id=item['id'], description=item['composition']['description'],
            object_specs=item['composition']['furniture'],
            extents=tuple(item['composition']['total_footprint'])
        ) for item in small_test_data
    ]
    
    debug_dir = "results_debug/scene_motif"
    output_dir = PROJECT_ROOT / Path(debug_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    def test_motif_generation(motifs, object_type):
        logger.info(f"Testing concurrent processing with {len(motifs)} motifs:")
        for motif in motifs:
            logger.info(f"  - {motif.id}: {motif.description}")
        logger.info("")
        
        asyncio.run(batch_inference(motifs, output_dir, "living_room", object_type=object_type, skip_visual_validation=skip_visual_validation, force_make_tight=force_make_tight)) 
    
    # test_motif_generation(test_large_motifs, ObjectType.LARGE)
    test_motif_generation(test_small_motifs, ObjectType.SMALL)

if __name__ == "__main__":
    main()