"""
Retrieval-Augmented Generation (RAG) module for the TrackRealties AI Platform.

This module provides intelligent search and context-aware responses for real estate data.
Enhanced with robust entity extraction and improved search routing.
"""

# Validation components
from ..validation.hallucination import RealEstateHallucinationDetector
# Enhanced entity extraction and routing (NEW)
from .llm_entity_extraction import EntityExtractionResult
from .llm_entity_extraction import \
    LLMEntityExtractor as RealEstateEntityExtractor
from .llm_entity_extraction import \
    extract_real_estate_entities_llm as extract_real_estate_entities
# Tools and utilities
# Core pipeline components
from .optimized_pipeline import EnhancedRAGPipeline, GreetingDetector
from .pipeline import RAGPipeline
from .router import IntelligentQueryRouter, QueryRouter
# Search components
from .search import GraphSearch, HybridSearchEngine, VectorSearch
# Legacy compatibility imports (for existing code)
from .smart_search import (FixedGraphSearch, QueryClassifier, QueryIntent,
                           QueryIntentClassifier, SearchStrategy,
                           SmartSearchRouter)
from .synthesizer import ResponseSynthesizer
from .tools import GraphSearchTool, VectorSearchTool

__all__ = [
    # Core Pipeline
    "EnhancedRAGPipeline",
    "RAGPipeline",
    "ResponseSynthesizer",
    # Search Components
    "VectorSearch",
    "GraphSearch",
    "HybridSearchEngine",
    "QueryRouter",
    "IntelligentQueryRouter",
    # Enhanced Entity Extraction (NEW)
    "RealEstateEntityExtractor",  # Now points to LLMEntityExtractor
    "EntityExtractionResult",
    "USStateMappings",
    "ExtractedEntity",
    "EntityType",
    "extract_real_estate_entities",  # Now points to LLM version
    # Smart Search Routing
    "SmartSearchRouter",
    "QueryIntentClassifier",
    "QueryClassifier",
    "SearchStrategy",
    "QueryIntent",
    # Tools
    "VectorSearchTool",
    "GraphSearchTool",
    "GreetingDetector",
    # Validation
    "RealEstateHallucinationDetector",
    "FixedGraphSearch",
]
