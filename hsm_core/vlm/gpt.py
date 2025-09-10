from __future__ import annotations
import base64
import io
import os
import pathlib
from typing import Any, Callable, List, Union
import datetime
import json
import hashlib

from matplotlib.figure import Figure
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from hsm_core.config import PROJECT_ROOT

MODEL: str = "gpt-4o-2024-08-06"
# MODEL = "o4-mini"
# MODEL = "gpt-4.1-2025-04-14"
# MODEL: str = "gpt-5"

REASONING_MODELS: list[str] = ["o3-mini", "o4-mini", "gpt-5"]
RETRY_COUNT: int = 5
MAX_IMAGE_SIZE: int = 2048

class Session:
    SESSION_OUTPUT_DIR = ""
    
    @classmethod
    def set_global_output_dir(cls, dir_path):
        """Set a global output directory for all Session instances"""
        cls.SESSION_OUTPUT_DIR = dir_path
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def __init__(self, prompts_path, model=MODEL, temperature: float = 0.7, output_dir: str = "", prompt_info: dict[str, str] | None = None) -> None:
        """
        Initialize a Session.
        """
        if not output_dir and Session.SESSION_OUTPUT_DIR:
            output_dir = Session.SESSION_OUTPUT_DIR
            
        load_dotenv()
        client = OpenAI()

        from hsm_core.utils import get_logger
        self.logger = get_logger('vlm.session')

        self.prompts_dir = prompts_path
        self.predefined_prompts = self._load_prompts(prompts_path)
        
        # Replace placeholders in the system prompt with the provided prompt_info
        if prompt_info and "system" in self.predefined_prompts.keys():
            system_prompt = self.predefined_prompts["system"]
            for key, value in prompt_info.items():
                placeholder = f"<{key.upper()}>"
                if placeholder in system_prompt:
                    system_prompt = system_prompt.replace(placeholder, str(value))
                    self.logger.debug(f"Initialized system prompt with {key}: {value}")
                else:
                    self.logger.warning(f"Placeholder {placeholder} not found in system prompt")
            self.predefined_prompts["system"] = system_prompt
            
        self.client = client
        self.model = model
        self.past_tasks: list[str] = []
        self.past_messages = [{"role": "system", "content": self.predefined_prompts["system"]}] 
        self.past_responses: list[str] = []
        self.temperature = temperature
        self.output_dir = output_dir

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens_this_session = 0
        
        prompts_path = pathlib.Path(self.prompts_dir)
        self._session_filename = f"{self.output_dir}/session_{prompts_path.stem}.txt"
        
    def _load_prompts(self, prompts_file: str) -> dict[str, str]:
        """
        Load predefined prompts from a YAML file.

        Args:
            prompts_file: str, path to the YAML file

        Returns:
            dict[str, str]: Dictionary of predefined prompts
        """
        with open(prompts_file) as file:
            return yaml.safe_load(file)
    
    def add_feedback(self, feedback: str) -> None:
        """
        Add feedback to the conversation for retry attempts.

        Args:
            feedback: The feedback message to add to the conversation.
        """
        retry_message = f"{feedback} Please try again."
        self.past_responses.append(retry_message)

    def reset_context(self) -> None:
        """
        Reset the conversation context by clearing past messages and responses.
        """
        self.past_messages = [{"role": "system", "content": self.predefined_prompts["system"]}]
        self.past_responses = []
    
    def send(self, task: str, prompt_info: dict[str, str] | None = None, 
             info_validate: bool = True, is_json: bool = False, verbose: bool = False, 
             images: Union[str, Figure, List[Union[str, Figure]], None] = None,
             image_detail: str = "high", append_text: str = "") -> str:
        """
        Send a message of a specific task to the VLM model and return the response.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            info_validate: boolean, whether to validate the input info
            json: boolean, whether the response should be in JSON format
            verbose: boolean, whether to print the prompt
            images: string, Figure, or list of them, the image(s) to be sent to the model
            image_detail: string, the detail level of the image
            append_text: string, additional text to append to the prompt
            use_cached: boolean, whether to use a cached response if available
            
        Returns:
            response: string, the response from the model
        """

        self.logger.debug(f"Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info, info_validate)
        if append_text:
            prompt = append_text + "\n\n" + prompt
        
        if images is not None:
            if isinstance(images, list):
                num_images = len(images)
            else:
                num_images = 1
        else:
            num_images = 0
        self.logger.debug(f"Past messages: {len(self.past_messages)} Prompt length: {len(prompt)} with {num_images} images")
        if verbose:
            self.logger.debug(f"Prompt:\n{prompt}")
        self._send(prompt, is_json, images, image_detail)
        response = self.past_responses[-1]
        if verbose:
            self.logger.debug(f"Response:\n{response}")

        return response
    
    def send_with_validation(self, task: str, 
                             prompt_info: dict[str, Any] | None = None, 
                             validation: Callable[[Any], tuple[bool, str, int]] | None = None,
                             retry: int = RETRY_COUNT,
                             images: Union[str, Figure, List[Union[str, Figure]], None] = None,
                             image_detail: str = "high",
                             is_json: bool = False,
                             verbose: bool = False) -> str:
        """
        Send a message of a specific task and return the response after validating it.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            validation: function, the validation function to validate the response for the task
            retry: integer, the number of retries for the task
            images: string, Figure, or list of them, the image(s) to be sent to the model
            image_detail: string, the detail level of the image
            json: boolean, whether the response should be in JSON format
            verbose: boolean, whether to print the prompt
        
        Returns:
            response: string, the response from the model
        """
        
        response = self.send(task, prompt_info, images=images, image_detail=image_detail, 
                          is_json=is_json, verbose=verbose)
        
        count = 0
        while count <= retry:
            if validation is not None:
                valid, error_message, error_index = validation(response)

                if not valid:
                    self.logger.info(f"Validation failed for task {task} at try {count+1}")
                    # Skip logging WN synset key validation errors
                    if error_message and not error_message.startswith("The WordNet synset key"):
                        self.logger.info(f"Validation error: {error_message}")

                    if count < retry:
                        count += 1
                        self.logger.info(f"Retrying task {task} [try {count+1} / {retry}]")
                        # Get the specific retry prompt if available (by order of appearance in the prompts file)
                        retry_prompt_keys = [key for key in self.predefined_prompts.keys() if task in key and "feedback" in key]
                        if retry_prompt_keys:
                            retry_task_name = retry_prompt_keys[error_index]
                        else:
                            # If there is no specific retry prompt, use the generic one
                            retry_task_name = "invalid_response"

                        response = self.send(retry_task_name, {"feedback": error_message}, is_json=is_json)
                        # Continue to next iteration for validation
                    else:
                        # No more retries available
                        break
                else:
                    self.logger.info(f"Validation passed for task {task} at try {count+1}")
                    break
            else:
                break  # No validation function, assume valid

        if count >= retry:
            raise RuntimeError(f"$ --- Validation failed for task {task} after {retry} retries")

        return response
    
    def _make_prompt(self, task: str, prompt_info: dict[str, str] | None, info_validate: bool = True) -> str:
        """
        Make a prompt for the VLM model.

        Args:
            task: string, the task of the prompt
            prompt_info: dictionary, the extra information for making the prompt for the task
            info_validate: boolean, whether to validate the input info
            
        Returns:
            prompt: string, the prompt for the VLM model
        """

        # Get the predefined prompt for the task
        prompt = self.predefined_prompts[task]

        # Check for task-specific required information
        valid = True
        match task:
            case "wnsynsetkey":
                valid = all(key in prompt_info for key in ["obj_label", "wnsynsetkeys"])
            case "wnsynsetkeys":
                valid = all(key in prompt_info for key in ["wnsynsetkeys", "object_labels"])
            
        if not valid and info_validate:
            raise ValueError(f"Extra information is required for the task: {task}")

        # Replace the placeholders in the prompt with the information
        if prompt_info is not None:
            for key in prompt_info:
                prompt = prompt.replace(f"<{key.upper()}>", str(prompt_info[key]))

        return prompt
    
    def _encode_image(self, image_or_path, detail="auto"):
        """Encode image for VLM models"""
        if isinstance(image_or_path, str):
            img = Image.open(image_or_path)
        elif isinstance(image_or_path, Figure):
            buf = io.BytesIO()
            image_or_path.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
        else:
            self.logger.debug(f"Unsupported image type: {type(image_or_path)}, value: {image_or_path}")
            raise ValueError(f"Warning: Unsupported image type: {type(image_or_path)}. Please provide a file path or a matplotlib Figure.")

        # Optimize image size based on detail level
        if detail == "low":
            target_size = (512, 512)
        elif detail == "high":
            target_size = (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)
        else:
            width, height = img.size
            if width * height <= 512 * 512:
                target_size = (512, 512)
            else:
                target_size = (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)

        # Preserve aspect ratio while resizing
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # removes alpha channel (Ref: https://www.oranlooney.com/post/gpt-cnn/)
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _send(self, new_message: str, json: bool = False, 
              images: Union[str, Figure, List[Union[str, Figure]], None] = None,
              image_detail="high", timeout: float = 15.0) -> None:
        """Send message to VLM models with optimized image handling and timeout."""
        message_content = []
        
        # Add text content first
        if new_message.strip():
            message_content.append({
                "type": "text",
                "text": new_message
            })
        
        # Handle multiple images
        if images is not None:
            # Convert single image to list for uniform processing and filter out None values
            image_list_raw = images if isinstance(images, list) else [images]
            image_list = [img for img in image_list_raw if img is not None]
            
            for image in image_list:
                try:
                    image_base64 = self._encode_image(image, detail=image_detail)
                except ValueError as exc:
                    self.logger.warning(
                        "Skipping unsupported image in _send (type=%s): %s", type(image), exc
                    )
                    continue
                if image_base64 is None:
                    continue

                message_content.append({
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": image_detail
                }
            })

        self.past_messages.append({"role": "user", "content": message_content})
        
        retries = 0
        max_retries = 3
        while retries < max_retries:
            params = {
                "model": self.model,
                "messages": self.past_messages,
                "response_format": { "type": "json_object" } if json else None,
                "temperature": self.temperature if self.model not in REASONING_MODELS else 1.0
            }

            if self.model in REASONING_MODELS:
                params["reasoning_effort"] = "high"
            
            completion = self.client.chat.completions.create(**params)
            response = completion.choices[0].message.content
            
            if completion.usage:
                self.total_prompt_tokens += completion.usage.prompt_tokens
                self.total_completion_tokens += completion.usage.completion_tokens
                self.total_tokens_this_session += completion.usage.total_tokens
            
            if response is not None:
                self.past_messages.append({"role": "assistant", "content": response})
                self.past_responses.append(response)
                return

            self.logger.info(f"Received None response, retrying... (Attempt {retries + 1}/{max_retries})")
            retries += 1
            
        raise RuntimeError(f"Failed to get a valid response after {max_retries} attempts")

    def save_session(self, filename=None):
        """
        Save the current session to a JSON file.
        
        Args:
            filename: Optional string, custom filename to save the session.
                     If None, uses the default session filename with .json extension.
        
        Returns:
            str: Path to the saved session file
        """
        if self.output_dir:
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
        if filename is None:
            prompts_path = pathlib.Path(self.prompts_dir)
            filename = f"session_{prompts_path.stem}.json"
            output_dir = self.output_dir if self.output_dir else "."
            filename = os.path.join(output_dir, filename)
        
        dir_path = os.path.dirname(os.path.abspath(filename))
        os.makedirs(dir_path, exist_ok=True)
        
        session_data = {
            "token_usage": self.get_session_token_usage(),
            "model": self.model,
            "temperature": self.temperature,
            "timestamp": datetime.datetime.now().isoformat(),
            "messages": [],
        }
        
        # Track prompt hashes we've seen to avoid duplicates due to retries
        processed_prompts = set()
        
        # Process messages in reverse order to find the latest valid response for each prompt
        tasks_with_responses = list(zip(self.past_tasks, self.past_responses))
        
        for i, (task, response) in enumerate(tasks_with_responses):
            # Get the corresponding user message
            if i * 2 + 1 >= len(self.past_messages):
                continue
                
            user_msg = self.past_messages[i * 2 + 1] if i > 0 else self.past_messages[1]
            assistant_msg = self.past_messages[i * 2 + 2] if i > 0 else self.past_messages[2]
            
            #TODO: handle image_url
            # stripped_content = [
            #     item for item in user_msg["content"]
            #     if item.get("type") != "image_url"
            # ]
            # content_str = str(stripped_content) 
            prompt_hash = hashlib.md5(str(user_msg["content"]).encode()).hexdigest()[:8]
            
            # Only save the latest response for each unique prompt
            if prompt_hash not in processed_prompts:
                processed_prompts.add(prompt_hash)
                
                exchange = {
                    "task": task,
                    "prompt_hash": prompt_hash,
                    "user_message": user_msg,
                    "assistant_message": assistant_msg,
                    "response": response
                }
                session_data["messages"].append(exchange)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
            
        # Session saved successfully
        return filename

    def get_session_token_usage(self):
        """
        Get the accumulated token usage for the current session.
        
        Returns:
            dict: A dictionary containing total_prompt_tokens, 
                  total_completion_tokens, and total_tokens_this_session.
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens_this_session": self.total_tokens_this_session
        }

def extract_program(response: str, description: str) -> Program:
    """
    Extract the program from the response of the VLM.

    Args:
        response: string, the response from the VLM
        description: string, the description of the program

    Returns:
        program: Program, the program extracted from the response
    """
    from hsm_core.scene_motif.programs.program import Program

    if "```python" in response:
        response = response.split("```python\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    code = response.split("\n")
    program = Program(code, description)

    return program

def extract_code(response: str) -> str:
    """
    Extract the code from the response of the VLM.

    Args:
        response: string, the response from the VLM
    
    Returns:
        code: string, the code extracted from the response
    """

    if "```python" in response:
        response = response.split("```python\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response

def extract_json(response: str) -> str:
    """
    Extract the JSON string from the response of the VLM.

    Args:
        response: string, the response from the VLM
        
    Returns:
        str: The extracted JSON string
    """

    if "```json" in response:
        response = response.split("```json\n")[1]
        response = response.split("```")[0]
    
    response = response.rstrip()

    return response