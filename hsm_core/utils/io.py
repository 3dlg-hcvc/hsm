import sys
from pathlib import Path
from datetime import datetime
from typing import TextIO, Tuple, Optional

class Logger:
    """Handles logging to both console and file with tqdm support."""
    
    def __init__(self, filename: Path) -> None:
        """
        Initialize logger with output file.
        
        Args:
            filename: Path to the main log file
        """
        self.terminal_stdout: TextIO = sys.__stdout__  # Use original stdout
        self.terminal_stderr: TextIO = sys.__stderr__  # Use original stderr
        self.log = open(filename, 'w', encoding='utf-8')
        self._closed = False
        
    def write(self, message: str) -> None:
        """
        Write a message to both the terminal and the log file.
        
        Args:
            message: The message to write
        """
        if not self._closed:
            # Always write to terminal first
            self.terminal_stdout.write(message)
            self.terminal_stdout.flush()
            
            # Write to log file
            self.log.write(message)
        
    def flush(self) -> None:
        """
        Flush both the terminal and the log file buffers.
        """
        if not self._closed:
            self.terminal_stdout.flush()
            self.log.flush()
        
    def close(self) -> None:
        """
        Close the log file.
        """
        if not self._closed:
            self.log.close()
            self._closed = True
            
    def __enter__(self) -> 'Logger':
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

class StderrLogger:
    """Handles stderr logging to both console and the main log file."""
    
    def __init__(self, logger: Logger) -> None:
        """
        Initialize stderr logger.
        
        Args:
            logger: Main logger instance
        """
        self.terminal_stderr: TextIO = sys.__stderr__  # Use original stderr
        self.logger = logger
        
    def write(self, message: str) -> None:
        """
        Write error message to console and log file.
        
        Args:
            message: The error message to write
        """
        # Write to terminal first
        self.terminal_stderr.write(message)
        self.terminal_stderr.flush()
        
        # Write to log file
        if not self.logger._closed:
            self.logger.log.write(message)
        
    def flush(self) -> None:
        """
        Flush console and log file buffers.
        """
        self.terminal_stderr.flush()
        if not self.logger._closed:
            self.logger.log.flush()

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

    # Use config value if available
    if cfg and hasattr(cfg, 'paths') and hasattr(cfg.paths, 'base_output_dir'):
        base_dir = cfg.paths.base_output_dir

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

def setup_logging(output_dir: Path) -> Tuple[Logger, StderrLogger]:
    """
    Set up logging for the application.
    
    Args:
        output_dir: Directory for log files
        
    Returns:
        tuple: (Logger, StderrLogger) instances
    """
    log_filename = output_dir / 'scene_log.log'
    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)
    
    logger = Logger(log_filename)
    stderr_logger = StderrLogger(logger)
    
    # Redirect stdout and stderr
    sys.stdout = logger
    sys.stderr = stderr_logger
    
    return logger, stderr_logger 