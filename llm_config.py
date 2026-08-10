"""Shared OpenAI-compatible LLM configuration.

Set DEEPSEEK_API_KEY to use DeepSeek V4 Flash.  The older OPENAI_API_*
variables remain supported for existing deployments.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple


def get_llm_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(api_key, base_url, model)`` for the configured judge."""
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return (
            deepseek_key,
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        )

    return (
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENAI_API_BASE"),
        os.getenv("OPENAI_API_MODEL"),
    )


def get_llm_timeout() -> float:
    """Bound every API request so a weekly run cannot hang indefinitely."""
    return float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))


def cache_namespace(prefix: str, prompt_version: str) -> str:
    """Isolate disk caches by model, endpoint, and prompt revision."""
    _, base_url, model = get_llm_config()
    raw = f"{prefix}_{model or 'legacy'}_{base_url or 'default'}_{prompt_version}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
