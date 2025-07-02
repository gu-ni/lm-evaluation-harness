from __future__ import annotations

from typing import Callable, Dict, List, Optional


def default_chat_template(
    messages: List[Dict[str, str]], add_generation_prompt: bool = True
) -> str:
    """A simple OpenAI-style chat template used as fallback."""
    chat = ""
    for m in messages:
        chat += f"<|{m['role']}|>{m['content']}"
    if add_generation_prompt:
        chat += "<|assistant|>"
    return chat


# Model specific descriptions
qwen25_math_description = (
    """You are a helpful assistant specialized in solving math problems."""
)
qwen2_math_description = """You are a helpful assistant specialized in mathematics."""
deepseek_math_description = """You are a math tutor that explains step by step."""

gemma_9b_description = """You are a helpful assistant."""
gemma_27b_description = """You are a helpful assistant."""


# Chat template examples (can be replaced with model specific ones)
def qwen_chat_template(
    messages: List[Dict[str, str]], add_generation_prompt: bool = True
) -> str:
    chat = ""
    for m in messages:
        role = m["role"]
        chat += f"<|im_start|>{role}\n{m['content']}<|im_end|>"
    if add_generation_prompt:
        chat += "<|im_start|>assistant\n"
    return chat


def gemma_chat_template(
    messages: List[Dict[str, str]], add_generation_prompt: bool = True
) -> str:
    chat = ""
    for m in messages:
        chat += f"<|start|>{m['role']}|{m['content']}<|end|>"
    if add_generation_prompt:
        chat += "<|start|>assistant|"
    return chat


MODEL_PROMPT_CONFIGS: Dict[str, Dict[str, object]] = {
    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "description": qwen25_math_description,
        "system_instruction": qwen25_math_description,
        "apply_chat_template": True,
        "chat_template": qwen_chat_template,
        "gen_prefix": "",
    },
    "Qwen/Qwen2-Math-7B-Instruct": {
        "description": qwen2_math_description,
        "system_instruction": qwen2_math_description,
        "apply_chat_template": True,
        "chat_template": qwen_chat_template,
        "gen_prefix": "",
    },
    "deepseek-math-7b-instruct": {
        "description": deepseek_math_description,
        "system_instruction": deepseek_math_description,
        "apply_chat_template": True,
        "chat_template": default_chat_template,
        "gen_prefix": "",
    },
    "google/gemma-2-9b-it": {
        "description": gemma_9b_description,
        "system_instruction": gemma_9b_description,
        "apply_chat_template": True,
        "chat_template": gemma_chat_template,
        "gen_prefix": "<bos>",
    },
    "google/gemma-2-27b-it": {
        "description": gemma_27b_description,
        "system_instruction": gemma_27b_description,
        "apply_chat_template": True,
        "chat_template": gemma_chat_template,
        "gen_prefix": "<bos>",
    },
    "nvidia/AceReason-Nemotron-1.1-7B": {
        "description": gemma_27b_description,
        "system_instruction": gemma_27b_description,
        "apply_chat_template": True,
        "chat_template": gemma_chat_template,
        "gen_prefix": "<bos>",
    }
}


def get_prompt_config(model_name: str) -> Optional[Dict[str, object]]:
    model_name = model_name.lower()
    for key, cfg in MODEL_PROMPT_CONFIGS.items():
        if key.lower() in model_name:
            return cfg
    return None
