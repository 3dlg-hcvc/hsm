from pathlib import Path
import numpy as np
import trimesh
from compress_json import compress_json
from dotenv import load_dotenv
from typing import Optional, Tuple, Dict, Any
from hsm_core.utils import get_logger

load_dotenv()
from hsm_core.config import HSSD_PATH
SUPPORT_DIR = HSSD_PATH / "support-surfaces"
logger = get_logger('support_region.loader')

def check_support_json_exists(id: str) -> bool:
    if not SUPPORT_DIR:
        logger.error("SUPPORT_DIR is not configured.")
        return False
    json_path = SUPPORT_DIR / id / f"{id}.supportSurface.json.gz"
    return json_path.exists()

def load_support_surface(id: str) -> Optional[Tuple[trimesh.Scene, Dict[str, Any], str]]:
    """Load support surface data from JSON file."""
    if not check_support_json_exists(id):
        logger.warning(f"Support surface JSON file for {id} does not exist")
        return None, None, None
    
    if not SUPPORT_DIR:
        logger.error("SUPPORT_DIR is not configured.")
        return None, None, None

    json_path = SUPPORT_DIR / id / f"{id}.supportSurface.json.gz"
    mesh_path = SUPPORT_DIR / id / f"{id}.supportSurface.glb"
    image_path = str(SUPPORT_DIR / id / f"{id}.supportSurface.png")
    
    try:
        scene = trimesh.load(mesh_path, force='scene')
        data = compress_json.load(str(json_path))
    except Exception as e:
        logger.error(f"Failed to load support surface data for {id}: {e}", exc_info=True)
        return None, None, None

    logger.debug(f"Loading support mesh: {mesh_path}")
    logger.debug(f"Loading support surface data from: {json_path}")
    
    return scene, data, image_path

def parse_support_surface(support_surface_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse support surface data into a structured format."""
    processed_data = {
        'id': support_surface_data['id'],
        'modelToWorld': np.array(support_surface_data['modelToWorld']).reshape(4, 4),
        'bbox': {
            'min': np.array(support_surface_data['bbox']['min']),
            'max': np.array(support_surface_data['bbox']['max'])
        },
        'surfaces': [],
        'modelbbox': {
            'min': np.array(support_surface_data.get('modelbbox', {}).get('min', [0,0,0])),
            'max': np.array(support_surface_data.get('modelbbox', {}).get('max', [0,0,0]))
        }
    }
    
    for surface in support_surface_data['supportSurfaces']:
        centroid = np.array(surface['obb']['centroid'])
        normal = np.array([surface['normal']['x'], surface['normal']['y'], surface['normal']['z']])
        normalized_axes = np.array(surface['obb']['normalizedAxes']).reshape(3, 3)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = normalized_axes
        
        surface_data = {
            'centroid': centroid,
            'normal': normal,
            'axes_lengths': np.array(surface['obb']['axesLengths']),
            'rotation_matrix': rotation_matrix,
            'area': surface['area'],
            'is_vertical': surface.get('isVertical', False),
            'is_horizontal': surface.get('isHorizontal', False),
            'is_interior': surface.get('isInterior', False),
            'mesh_index': surface.get('meshIndex', 0),
            'mesh_face_indices': surface.get('meshFaceIndices', []),
            'samples': [],
            'obb_metadata': surface['obb'].get('metadata', {})
        }
        
        processed_samples = []
        for sample in surface.get('samples', []):
            sample_data = {
                'point': np.array(sample['point']),
                'normal': np.array(sample['normal']),
                'model_point': np.array(sample['modelPoint']),
                'model_normal': np.array(sample['modelNormal']),
                'uv': np.array(sample['uv']) if 'uv' in sample else None,
                'clearance': sample.get('clearance', float('inf'))
            }
            processed_samples.append(sample_data)
            
        surface_data['samples'] = processed_samples
        processed_data['surfaces'].append(surface_data)
    
    return processed_data 