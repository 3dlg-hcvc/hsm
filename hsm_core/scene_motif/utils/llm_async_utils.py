import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


async def send_llm_async(session, task: str, prompt_info: dict, **kwargs) -> str:
    """Async wrapper for synchronous VLM calls."""
    return await asyncio.to_thread(session.send, task, prompt_info, **kwargs)


async def send_llm_with_validation_async(session, task: str, prompt_info: dict, validation_fn: Callable, **kwargs) -> str:
    """Async wrapper for synchronous VLM calls with validation."""
    return await asyncio.to_thread(session.send_with_validation, task, prompt_info, validation_fn, **kwargs)


async def send_llm_with_images_async(session, task: str, prompt_info: dict, images: Any, **kwargs) -> str:
    """Async wrapper for synchronous VLM calls with images."""
    return await asyncio.to_thread(session.send, task, prompt_info, images=images, **kwargs)
