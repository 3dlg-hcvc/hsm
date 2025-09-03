from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PROMPT_DIR = PROJECT_ROOT / "configs" / "prompts"
DATA_PATH = PROJECT_ROOT / "data"

# HSSD (Habitat Synthetic Scene Dataset) models path
# This should point to the directory containing HSSD model data
HSSD_PATH = DATA_PATH / "hssd-models"

if __name__ == "__main__":
    print(PROJECT_ROOT)