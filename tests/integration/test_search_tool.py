# tests/integration/test_search_tool.py

import pytest
from src.search_tool import create_search_tool
from src.config import config
from src.exceptions import SearchToolError
import asyncio


@pytest.fixture
async def search_tool():
    return await create_search_tool(config, config.serper_api_key.get_secret_value())


@pytest.mark.asyncio
async def test_search_tool_creation(search_tool):
    assert search_tool is not None
    assert search_tool.name == "Search"
    assert callable(search_tool.func)


@pytest.mark.asyncio
async def test_search_successful_query(search_tool):
    query = "What is the capital of France?"
    result = await search_tool.func(query)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "Paris" in result.lower()


@pytest.mark.asyncio
async def test_search_empty_query(search_tool):
    query = ""
    result = await search_tool.func(query)

    assert isinstance(result, str)
    assert len(result) > 0  # Even for an empty query, we expect some kind of result


@pytest.mark.asyncio
async def test_search_complex_query(search_tool):
    query = "What are the top 3 programming languages in 2023?"
    result = await search_tool.func(query)

    assert isinstance(result, str)
    assert len(result) > 0
    # Check for common programming languages
    assert any(
        lang.lower() in result.lower()
        for lang in ["python", "javascript", "java", "c++", "go"]
    )


@pytest.mark.asyncio
async def test_search_with_special_characters(search_tool):
    query = "What is the meaning of 'λ' in physics?"
    result = await search_tool.func(query)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "wavelength" in result.lower() or "lambda" in result.lower()


@pytest.mark.asyncio
async def test_search_rate_limiting():
    search_tool = await create_search_tool(
        config, config.serper_api_key.get_secret_value()
    )
    queries = ["python", "javascript", "java", "c++", "go"]

    results = await asyncio.gather(*(search_tool.func(query) for query in queries))

    assert len(results) == len(queries)
    for result in results:
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_search_invalid_api_key():
    with pytest.raises(SearchToolError):
        await create_search_tool(config, "invalid_api_key")


@pytest.mark.asyncio
async def test_search_network_error(search_tool):
    # Simulate a network error by providing an invalid URL
    original_url = search_tool.search.url
    search_tool.search.url = "https://invalid-url.com"

    with pytest.raises(SearchToolError):
        await search_tool.func("test query")

    # Restore the original URL
    search_tool.search.url = original_url


@pytest.mark.asyncio
async def test_search_result_processing(search_tool):
    query = "Who won the last FIFA World Cup?"
    result = await search_tool.func(query)

    assert isinstance(result, str)
    assert len(result) > 0
    assert (
        len(result) <= config.search_result_limit
    )  # Check if result is within the configured limit


# Helper function to run multiple searches concurrently
async def run_concurrent_searches(search_tool, queries):
    return await asyncio.gather(*(search_tool.func(query) for query in queries))


@pytest.mark.asyncio
async def test_concurrent_searches(search_tool):
    queries = [
        "What is the population of New York?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the boiling point of water?",
        "When was the first iPhone released?",
        "What is the largest planet in our solar system?",
    ]

    results = await run_concurrent_searches(search_tool, queries)

    assert len(results) == len(queries)
    for result in results:
        assert isinstance(result, str)
        assert len(result) > 0
