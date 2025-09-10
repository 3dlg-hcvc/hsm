from pathlib import Path
from datetime import datetime
from typing import Optional

def create_output_directory(base_dir: str = "results", subfix: str = "", timestamp: bool = True, project_root: Optional[Path] = None, cfg: Optional[object] = None) -> Path:
    """
    Create output directory with timestamp relative to project root.

    Args:
        base_dir: Base directory name (default: "results" to match config)
        subfix: Optional suffix for directory name
        timestamp: Whether to include a timestamp in the directory name
        project_root: Project root path. If None, uses PROJECT_ROOT from constants
        cfg: Configuration object to get base_dir from paths section

    Returns:
        Path: Created directory path
    """
    from hsm_core.config import PROJECT_ROOT

    root = project_root if project_root is not None else PROJECT_ROOT
    ts = datetime.now().strftime("%m%d-%H%M") if timestamp else ""

    if ts:
        output_dir = root / base_dir / f"{ts}_{subfix}"
    else:
        output_dir = root / base_dir / subfix

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_current_iteration(folder_path: Path) -> int:
    """
    Get current iteration number for output directory.
    
    Args:
        folder_path: Base folder path
        
    Returns:
        int: Next iteration number
    """
    iteration_folders = list(folder_path.parent.glob(f"{folder_path.name}_iteration_*"))
    
    if not iteration_folders:
        return 1
        
    iteration_numbers = []
    for folder in iteration_folders:
        try:
            num = int(folder.name.split("_iteration_")[-1])
            iteration_numbers.append(num)
        except (ValueError, IndexError):
            continue
            
    return max(iteration_numbers, default=0)

def get_next_iteration(folder_path: Path) -> int:
    """
    Get next iteration number for output directory.
    
    Args:
        folder_path: Base folder path

    Returns:
        int: Next iteration number
    """
    return (get_current_iteration(folder_path) + 1)
