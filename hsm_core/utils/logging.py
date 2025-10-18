"""
HSM Logging
Uses short logger names by default (e.g., hsm_core.scene_motif instead of full paths).

Usage:
    import logging
    logger = logging.getLogger(__name__)

    logger.info("This works automatically with HSM's configured logging!")
    logger.debug("Debug messages")
    logger.warning("Warnings")
    logger.error("Errors")

    After calling setup_logging() once at application startup, all
    logging.getLogger(__name__) calls will automatically use HSM's
    custom formatting and handlers.
"""

from hsm_core.config import GLOBAL_LOGGING_LEVEL_THRESHOLD, LOGGING_LEVEL_TERMINAL, LOGGING_LEVEL_FILE

import logging
import sys
from pathlib import Path

class HSMFormatter(logging.Formatter):
    def format(self, record):
        # Convert full name to short name: hsm_core.module.submodule -> hsm_core.module
        if record.name.startswith('hsm_core.'):
            parts = record.name.split('.')
            if len(parts) >= 3:
                record.name = '.'.join(parts[:3])
        return super().format(record)

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Setup unified logging for HSM - everything goes to scene.log.
    Configures the logging system so that logging.getLogger(__name__) works automatically
    for all hsm_core modules without needing to import get_logger().

    Args:
        output_dir: Directory for log files

    Returns:
        Main logger instance
    """
    # Clear any existing handlers from root logger
    # root_logger = logging.getLogger()
    # for handler in root_logger.handlers[:]:
    #     root_logger.removeHandler(handler)

    # # Also clear handlers from any existing hsm_core logger
    # hsm_logger = logging.getLogger('hsm_core')
    # for handler in hsm_logger.handlers[:]:
    #     hsm_logger.removeHandler(handler)

    # Custom formatter for short names
    formatter = HSMFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler for main scene log
    log_file = output_dir / 'scene.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(LOGGING_LEVEL_FILE)
    file_handler.setFormatter(formatter)

    # Terminal handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOGGING_LEVEL_TERMINAL)
    console_handler.setFormatter(HSMFormatter('%(message)s'))

    # Configure main HSM logger - this will be the parent for all hsm_core loggers
    logger = logging.getLogger('hsm_core')
    logger.setLevel(GLOBAL_LOGGING_LEVEL_THRESHOLD)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Stop propagation to root to avoid duplicate terminal logs
    logger.propagate = False

    # Configure root logger to allow hsm_core loggers to work with getLogger(__name__)
    # root_logger.setLevel(logging.WARNING)

    # # Configure retrieval logger
    # retrieval_logger = logging.getLogger('hsm_core.retrieval')
    # retrieval_logger.setLevel(logging.WARNING)
    # retrieval_logger.propagate = False

    # # Configure VLM logger
    # vlm_logger = logging.getLogger('hsm_core.vlm')
    # vlm_logger.setLevel(logging.WARNING)
    # vlm_logger.propagate = False

    # Suppress trimesh warnings
    for ext_name, level in (
        ('trimesh', logging.ERROR),
        ('trimesh.util', logging.ERROR),
    ):
        ext_logger = logging.getLogger(ext_name)
        ext_logger.setLevel(level)
        # Avoid propagating to root to prevent accidental printing
        ext_logger.propagate = False

    return logger


def get_motif_logger(name: str, motif_id: str, motif_output_dir: Path) -> logging.Logger:
    """
    Get a logger for motif processing.
    """
    base_name = f"hsm_core.{name}" if not name.startswith('hsm_core.') else name
    return logging.getLogger(base_name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger.
    This function is kept for backward compatibility.
    """
    full_name = f"hsm_core.{name}" if not name.startswith('hsm_core.') else name
    logger = logging.getLogger(full_name)

    return logger
