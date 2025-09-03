import yaml
import torch
import numpy as np
from dotenv import load_dotenv


load_dotenv()
from hsm_core.config import PROJECT_ROOT
EMBEDDING_PATH = PROJECT_ROOT / "data" / "preprocessed"

def load_hssd_embeddings_and_index(embedding_model: str = "clip") -> tuple[torch.Tensor, list]:
    """
    Load the HSSD embeddings.

    Args:
        embedding_model: string, the name of the embedding model to use for retrieval ("clip" or "ddclip")
    
    Returns:
        embeddings: torch.Tensor, the HSSD embeddings
        index: list, the HSSD embeddings index
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_path = EMBEDDING_PATH / f"{embedding_model}_hssd_embeddings.npy"
    index_path = EMBEDDING_PATH / f"{embedding_model}_hssd_embeddings_index.yaml"

    # Load embeddings
    with open(embedding_path, "rb") as f:
        # Load as float32 to avoid dtype mismatch issues
        hssd_embeddings = torch.tensor(np.load(f), dtype=torch.float32).to(device)

    # Load index
    with open(index_path, "r") as f:
        hssd_embedding_index = yaml.safe_load(f)

    if embedding_model == "clip":
        hssd_embeddings /= hssd_embeddings.norm(dim=-1, keepdim=True)

    return hssd_embeddings, hssd_embedding_index




    