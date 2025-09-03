"""
VLM Factory Module

This module provides a unified interface for creating VLM sessions,
supporting both GPT (OpenAI) and other VLM models.
"""

from typing import Optional, Dict, Any
from pathlib import Path

def create_session(
    prompts_path: str, 
    model_type: str = "gpt", 
    model_name: str = "gpt-4o-2024-08-06",
    temperature: float = 0.7, 
    output_dir: str = "", 
    prompt_info: Dict[str, str] = {},
):
    """
    Factory function to create either GPT or Qwen session based on model_type.
    
    Args:
        prompts_path: Path to the YAML file containing prompts
        model_type: Either "gpt" or "qwen" to specify which session type to create
        model_name: Model name (e.g., "gpt-4o-2024-08-06" for GPT or "Qwen2.5-VL-7B-Instruct" for Qwen)
        temperature: Sampling temperature for the model
        output_dir: Directory to save session logs
        prompt_info: Optional dictionary to replace placeholders in the system prompt
        
    Returns:
        Session object (either GPT Session or QwenSession)
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == "gpt":
        try:
            from hsm_core.vlm.gpt import Session
            return Session(prompts_path, model=model_name, temperature=temperature, 
                          output_dir=output_dir, prompt_info=prompt_info)
        except ImportError as e:
            raise ImportError(f"Failed to import GPT Session: {e}")
    
    # elif model_type.lower() == "qwen":
    #     try:
    #         from hsm_core.vlm.qwen import QwenSession
    #         if model_name is None:
    #             model_name = "Qwen2.5-VL-7B-Instruct"
    #         return QwenSession(prompts_path, model=model_name, temperature=temperature, 
    #                           output_dir=output_dir, prompt_info=prompt_info, 
    #                           use_quantized=use_quantized, quantization_type=quantization_type)
    #     except ImportError as e:
    #         raise ImportError(f"Failed to import QwenSession: {e}")
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'gpt'")
        # raise ValueError(f"Unsupported model_type: {model_type}. Use 'gpt' or 'qwen'")

def create_gpt_session(prompts_path: str, **kwargs):
    """
    Convenience function to create a GPT session.
    
    Args:
        prompts_path: Path to the YAML file containing prompts
        **kwargs: Additional arguments passed to create_session
        
    Returns:
        GPT Session object
    """
    return create_session(prompts_path, model_type="gpt", **kwargs)

# def create_qwen_session(prompts_path: str, **kwargs):
#     """
#     Convenience function to create a Qwen session.
    
#     Args:
#         prompts_path: Path to the YAML file containing prompts
#         **kwargs: Additional arguments passed to create_session
        
#     Returns:
#         QwenSession object
#     """
#     return create_session(prompts_path, model_type="qwen", **kwargs)

# For backward compatibility - alias the main function
Session = create_session 