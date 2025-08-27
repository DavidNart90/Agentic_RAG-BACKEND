"""
Flexible provider configuration for LLM and embedding models.
"""

import os
from typing import Optional

import openai
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..core.config import get_settings

settings = get_settings()


def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get LLM model configuration based on environment variables.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured OpenAI-compatible model
    """
    llm_choice = model_choice or settings.DEFAULT_MODEL
    api_key = settings.OPENAI_API_KEY

    provider = OpenAIProvider(api_key=api_key)
    return OpenAIModel(llm_choice, provider=provider)


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get embedding client configuration based on environment variables.

    Returns:
        Configured OpenAI-compatible client for embeddings
    """
    return openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.

    Returns:
        Embedding model name
    """
    return settings.EMBEDDING_MODEL
