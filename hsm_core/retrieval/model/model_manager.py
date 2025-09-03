import asyncio
import torch
import threading
import multiprocessing
from typing import Tuple, Dict, Any
import open_clip
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer
import logging

MODEL_NAME = "ViT-H-14-378-quickgelu"
PRETRAINED = "dfn5b"

logger = logging.getLogger(__name__)


class ModelManager:
    _lock = threading.RLock()
    _process_id: int = 0
    _clip_model: Dict[int, Dict[str, Any]] = {}
    _clip_tokenizer: Dict[int, Dict[str, Any]] = {}
    _embedding_cache: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    _initialized = False
    _model_init_tasks: Dict[str, asyncio.Task] = {}

    @classmethod
    def initialize(cls):
        """Initialize ModelManager for the current process."""
        with cls._lock:
            if not cls._initialized:
                pid = multiprocessing.current_process().pid
                assert pid is not None, "Process ID should never be None"
                cls._process_id = pid
                cls._clip_model[cls._process_id] = {}
                cls._clip_tokenizer[cls._process_id] = {}
                cls._embedding_cache[cls._process_id] = {}
                cls._initialized = True
                logger.info(f"ModelManager initialized for process {cls._process_id}")

    @classmethod
    async def initialize_model_async(cls, model_name: str, pretrained: str) -> None:
        """Initialize a specific model asynchronously."""
        if not cls._initialized:
            cls.initialize()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_key = f"{model_name}_{pretrained}"

        # If model is already loaded, do nothing
        if model_key in cls._clip_model[cls._process_id]:
            return

        # Use thread-safe model loading
        try:
            logger.info(f"Loading model '{model_key}' for process {cls._process_id} on device {device}")

            loop = asyncio.get_event_loop()
            model, tokenizer = await asyncio.wait_for(
                loop.run_in_executor(None, cls._load_clip_model, model_name, pretrained, device),
                timeout=600
            )

            if model is None or tokenizer is None:
                raise RuntimeError(f"Model or tokenizer for '{model_key}' failed to load")

            cls._clip_model[cls._process_id][model_key] = model
            cls._clip_tokenizer[cls._process_id][model_key] = tokenizer
            cls._embedding_cache[cls._process_id][model_key] = {}
            logger.info(f"Model '{model_key}' loaded successfully for process {cls._process_id}")

        except Exception as e:
            logger.error(f"Error loading model '{model_key}': {str(e)}")
            # Cleanup on failure
            for storage in [cls._clip_model, cls._clip_tokenizer]:
                if cls._process_id in storage and model_key in storage[cls._process_id]:
                    del storage[cls._process_id][model_key]
            raise

    @classmethod
    async def get_clip_model_async(cls, model_name: str=MODEL_NAME, pretrained: str=PRETRAINED) -> Tuple[Any, Any]:
        """Get or initialize a specific OpenCLIP model and tokenizer."""
        if not cls._initialized:
            cls.initialize()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_key = f"{model_name}_{pretrained}"

        # Initialize model if not loaded
        if model_key not in cls._clip_model[cls._process_id]:
            if model_key not in cls._model_init_tasks:
                cls._model_init_tasks[model_key] = asyncio.create_task(
                    cls.initialize_model_async(model_name, pretrained)
                )
            await cls._model_init_tasks[model_key]
            del cls._model_init_tasks[model_key]

        if model_key not in cls._clip_model[cls._process_id]:
            raise RuntimeError(f"Model initialization failed for '{model_key}'")

        return cls._clip_model[cls._process_id][model_key].to(device), cls._clip_tokenizer[cls._process_id][model_key]

    @staticmethod
    def _load_clip_model(model_name: str, pretrained: str, device: str) -> Tuple[Any, Any]:
        """Helper method to load a specific CLIP model synchronously."""
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()  # type: ignore
            model = model.float()  # type: ignore
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error in _load_clip_model: {e}")
            raise

    @classmethod
    def clear_cache(cls, model_name: str=MODEL_NAME, pretrained: str=PRETRAINED):
        """Clear cache and model for a specific model in the current process."""
        model_key = f"{model_name}_{pretrained}"
        with cls._lock:
            pid = cls._process_id
            if model_key in cls._clip_model.get(pid, {}):
                del cls._clip_model[pid][model_key]
                del cls._clip_tokenizer[pid][model_key]
                cls._embedding_cache[pid].pop(model_key, None)
                logger.info(f"Cleared cache and resources for model '{model_key}'")