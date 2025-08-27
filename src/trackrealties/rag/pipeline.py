"""
RAG Pipeline for generating and validating responses.
"""

import logging
from typing import Dict

from src.trackrealties.rag.llm_entity_extraction import \
    LLMEntityExtractor as RealEstateEntityExtractor
from src.trackrealties.rag.search import HybridSearchEngine
from src.trackrealties.rag.synthesizer import ResponseSynthesizer
from src.trackrealties.rag.validation import ResponseValidator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the response generation process using RAG components.
    """

    def __init__(self):
        self.search_engine = HybridSearchEngine()
        self.synthesizer = ResponseSynthesizer()
        self.validator = ResponseValidator()
        self.entity_extractor = RealEstateEntityExtractor()

    async def generate_response(self, query: str, context: Dict) -> str:
        """
        Generates a response by searching for relevant information, synthesizing a
        draft, and validating the result.

        Args:
            query: The user's query.
            context: The analysis results from the services.

        Returns:
            The validated response.
        """
        # 1. Extract entities from the query
        entities = await self.entity_extractor.extract_entities(query)
        logger.info(f"Extracted entities: {entities}")

        # 2. Build filters from entities
        filters = context.get("filters", {})
        for entity in entities:
            if entity["type"] == "GPE":  # Geopolitical Entity (e.g., city, state, country)
                filters["location"] = entity["name"]

        logger.info(f"Search filters: {filters}")

        # 3. Search for relevant information
        search_results = await self.search_engine.search(query, limit=context.get("limit", 10), filters=filters)
        logger.info(f"Search results: {search_results}")

        # 4. Synthesize a draft response
        draft_response = await self.synthesizer.synthesize_response(query, search_results)

        # 5. Validate the response
        validated_response, _ = await self.validator.validate_response(draft_response, context, search_results)

        return validated_response
