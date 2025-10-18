"""
Consolidated Similarities Computation Module

This module provides unified similarity computation functionality for both local and server-based
retrieval systems, eliminating code duplication across the retrieval module.
"""

from typing import List, Tuple, Optional, Any
import torch
from hsm_core.utils import get_logger

from ..model.embeddings import load_hssd_embeddings_and_index

logger = get_logger('retrieval.utils.similarities')


def filter_hssd_embeddings(
    hssd_embeddings: torch.Tensor,
    hssd_index: List[str],
    filter_indices: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[torch.Tensor, List[str]]:
    """
    Filter HSSD embeddings based on provided mesh IDs.

    Args:
        hssd_embeddings: Full HSSD embeddings tensor
        hssd_index: Full HSSD index list
        filter_indices: Optional list of mesh IDs to filter by
        verbose: Whether to log filtering results

    Returns:
        Tuple of (filtered_embeddings, filtered_index)
    """
    if filter_indices is None:
        return hssd_embeddings, hssd_index

    # Create mapping for faster lookup
    valid_ids_set = set(filter_indices)
    hssd_ids_set = set(hssd_index)

    # Find intersection of valid IDs and available HSSD embeddings
    valid_ids = valid_ids_set.intersection(hssd_ids_set)

    if not valid_ids:
        if verbose:
            logger.info(f"None of the {len(valid_ids_set)} requested meshes found in HSSD embeddings")
        # Return empty tensor with correct shape
        return torch.zeros((0, hssd_embeddings.shape[1]), device=hssd_embeddings.device, dtype=hssd_embeddings.dtype), []

    # Get indices and mesh IDs in consistent order
    valid_indices = []
    filtered_mesh_ids = []
    for i, mesh_id in enumerate(hssd_index):
        if mesh_id in valid_ids:
            valid_indices.append(i)
            filtered_mesh_ids.append(mesh_id)

    filtered_embeddings = hssd_embeddings[valid_indices]

    if verbose:
        logger.info(f"Found {len(filtered_mesh_ids)} meshes in HSSD embeddings out of {len(valid_ids_set)} requested")

    return filtered_embeddings, filtered_mesh_ids


def compute_text_embeddings(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute text embeddings for a list of texts.

    Args:
        texts: List of text strings
        model: CLIP model
        tokenizer: CLIP tokenizer
        device: Device to use for computation

    Returns:
        Text embeddings tensor
    """
    if not texts:
        return torch.zeros((0, 512), device=device, dtype=torch.float32)  # CLIP embedding dimension

    tokenized_texts = tokenizer(texts).to(device)

    if device == "cuda":
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            text_embeddings = model.encode_text(tokenized_texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    else:
        with torch.no_grad():
            text_embeddings = model.encode_text(tokenized_texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    # Ensure output is float32 to match HSSD embeddings
    return text_embeddings.to(device=device, dtype=torch.float32)


def compute_similarities_batch(
    text_embeddings: torch.Tensor,
    hssd_embeddings: torch.Tensor,
    chunk_size: int = 1000
) -> torch.Tensor:
    """
    Compute similarities between text and HSSD embeddings in chunks.

    Args:
        text_embeddings: Text embeddings tensor
        hssd_embeddings: HSSD embeddings tensor
        chunk_size: Size of chunks for memory-efficient computation

    Returns:
        Similarities tensor
    """
    if text_embeddings.shape[0] == 0 or hssd_embeddings.shape[0] == 0:
        return torch.zeros((text_embeddings.shape[0], hssd_embeddings.shape[0]),
                          device=text_embeddings.device, dtype=text_embeddings.dtype)

    similarities_chunks = []
    for i in range(0, hssd_embeddings.shape[0], chunk_size):
        chunk_embeddings = hssd_embeddings[i:i+chunk_size]
        chunk_similarities = torch.matmul(text_embeddings, chunk_embeddings.T)
        similarities_chunks.append(chunk_similarities)

    return torch.cat(similarities_chunks, dim=1)

def compute_similarities(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    filter_indices: Optional[List[str]] = None,
    embedding_model: str = "clip",
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[torch.Tensor, List[str]]:
    """
    Unified function to compute similarities between texts and HSSD embeddings.

    Args:
        texts: List of text strings
        model: CLIP model
        tokenizer: CLIP tokenizer
        filter_indices: Optional list of mesh IDs to filter by
        embedding_model: Embedding model name ("clip" or "ddclip")
        device: Device for computation
        verbose: Whether to log progress

    Returns:
        Tuple of (similarities_tensor, mesh_ids_list)
    """
    try:
        if not texts:
            return torch.zeros((0, 0), device=device, dtype=torch.float32), []

        # Load HSSD embeddings
        hssd_embeddings, hssd_index = load_hssd_embeddings_and_index(embedding_model)

        # Filter embeddings if mesh IDs provided
        filtered_embeddings, filtered_index = filter_hssd_embeddings(
            hssd_embeddings, hssd_index, filter_indices, verbose
        )

        if filtered_embeddings.shape[0] == 0:
            return torch.zeros((len(texts), 0), device=device, dtype=torch.float32), []

        # Compute text embeddings
        text_embeddings = compute_text_embeddings(texts, model, tokenizer, device)

        # Ensure filtered embeddings are on the same device and dtype
        if filtered_embeddings.device != text_embeddings.device:
            filtered_embeddings = filtered_embeddings.to(device=device, dtype=torch.float32)

        # Compute similarities
        similarities = compute_similarities_batch(text_embeddings, filtered_embeddings)

        return similarities, filtered_index

    except Exception as e:
        logger.error(f"Error computing similarities: {e}")
        raise



