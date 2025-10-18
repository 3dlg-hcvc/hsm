from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PROMPT_DIR = PROJECT_ROOT / "configs" / "prompts"
DATA_PATH = PROJECT_ROOT / "data"

# HSSD (Habitat Synthetic Scene Dataset) models path
# This should point to the directory containing HSSD model data
HSSD_PATH = DATA_PATH / "hssd-models"

GLOBAL_LOGGING_LEVEL_THRESHOLD = logging.DEBUG
LOGGING_LEVEL_TERMINAL = logging.WARNING
LOGGING_LEVEL_FILE = logging.INFO

# if __name__ == "__main__":
    # logger = get_logger('hsm_core.config')
    # logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")