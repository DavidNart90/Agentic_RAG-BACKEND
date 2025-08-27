import asyncio

from src.trackrealties.rag.smart_search import SmartSearchRouter


async def test_smart_search_router() -> None:
    """Simple demonstration of the SmartSearchRouter."""
    router = SmartSearchRouter()

    test_queries = [
        "What is the median price in Dallas, TX?",
        "Should I invest in Austin real estate?",
        "Tell me about market trends in Texas",
        "Who is the agent for property 123-456?",
        "Compare Austin vs Dallas markets",
        "Find me properties under $400K in Houston",
    ]

    for query in test_queries:
        strategy = await router.route_search(query)
        print(f"Query: '{query}' -> Strategy: {strategy}")


if __name__ == "__main__":
    asyncio.run(test_smart_search_router())
