"""
This module defines the ResponseSynthesizer component for the TrackRealties AI Platform.

The ResponseSynthesizer is responsible for generating coherent and contextually
relevant responses based on the user's query and the search results retrieved
from the knowledge graph or other data sources.
"""

import logging
from typing import List

from pydantic_ai import Agent

from ..core.config import get_settings
from ..core.llm import get_llm_model
from ..models.search import SearchResult

settings = get_settings()
logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    """
    The ResponseSynthesizer is responsible for generating a coherent and contextually
    relevant response based on the user's query and the search results retrieved
    from the knowledge graph or other data sources.
    """

    def __init__(self):
        """Initialize the ResponseSynthesizer."""
        self.logger = logging.getLogger(__name__)
        llm = get_llm_model(model_choice=settings.DEFAULT_MODEL)
        self.agent = Agent(llm=llm)

    async def synthesize_response(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Generates a response by combining the search results into a single,
        formatted string.

        Args:
            query (str): The original query from the user.
            search_results (list): A list of SearchResult objects.

        Returns:
            str: A formatted response string.
        """
        normalized = query.lower().strip()
        greetings = {"hi", "hello", "hey", "hi agent", "hello agent", "hey agent"}
        if normalized in greetings:
            return "Hello! How can I assist you with real estate today?"

        if not search_results:
            return f"No relevant information found for your query: '{query}'"

        combined_results = "\n".join([result.content for result in search_results])

        prompt = f"""
        You are a real estate analysis assistant. Based on the user's query and the provided data, generate a concise and helpful response.

        User Query: {query}

        Data:
        {combined_results}
        """
        response = await self.agent.run(prompt)
        return response

    async def stream_response(self, query: str, search_results: List[SearchResult]):
        """
        Generates a streaming response by combining the search results.

        Args:
            query (str): The original query from the user.
            search_results (list): A list of SearchResult objects.
        """
        if not search_results:
            yield f"No relevant information found for your query: '{query}'"
            return

        combined_results = "\n".join([result.content for result in search_results])

        prompt = f"""
        You are a real estate analysis assistant. Based on the user's query and the provided data, generate a concise and helpful response.

        User Query: {query}

        Data:
        {combined_results}
        """
        async for chunk in self.agent.stream(prompt):
            yield chunk
