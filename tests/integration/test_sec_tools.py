# tests/integration/test_sec_tools.py

import pytest
import os
from src.sec_tools import SECTools
from src.config import config
from src.exceptions import SECToolsError, FilingNotFoundError, EmbeddingSearchError

@pytest.fixture
def sec_tools():
    return SECTools(config, config.sec_api_key.get_secret_value())

@pytest.mark.asyncio
async def test_search_10q_valid_ticker(sec_tools):
    result = await sec_tools.search_10q("AAPL|What was the revenue last quarter?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "revenue" in result.lower()

@pytest.mark.asyncio
async def test_search_10k_valid_ticker(sec_tools):
    result = await sec_tools.search_10k("MSFT|What was the annual revenue?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "revenue" in result.lower()

@pytest.mark.asyncio
async def test_search_10q_invalid_ticker(sec_tools):
    result = await sec_tools.search_10q("INVALID|What was the revenue?")
    assert "couldn't find any filing for this stock" in result

@pytest.mark.asyncio
async def test_search_10k_invalid_ticker(sec_tools):
    result = await sec_tools.search_10k("INVALID|What was the revenue?")
    assert "couldn't find any filing for this stock" in result

@pytest.mark.asyncio
async def test_search_10q_invalid_query_format(sec_tools):
    with pytest.raises(SECToolsError):
        await sec_tools.search_10q("AAPL")  # Missing the question part

@pytest.mark.asyncio
async def test_search_10k_invalid_query_format(sec_tools):
    with pytest.raises(SECToolsError):
        await sec_tools.search_10k("MSFT")  # Missing the question part

@pytest.mark.asyncio
async def test_search_10q_complex_query(sec_tools):
    result = await sec_tools.search_10q("GOOGL|What were the main risks mentioned in the last quarter?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert any(word in result.lower() for word in ["risk", "challenge", "uncertainty"])

@pytest.mark.asyncio
async def test_search_10k_complex_query(sec_tools):
    result = await sec_tools.search_10k("AMZN|What were the major acquisitions in the last year?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert any(word in result.lower() for word in ["acquisition", "purchase", "buyout"])

@pytest.mark.asyncio
async def test_search_rate_limiting(sec_tools):
    queries = [
        "AAPL|What was the revenue?",
        "MSFT|What were the operating expenses?",
        "GOOGL|What was the net income?",
        "AMZN|What were the total assets?",
        "FB|What was the earnings per share?"
    ]
    
    for query in queries:
        result = await sec_tools.search_10q(query)
        assert isinstance(result, str)
        assert len(result) > 0

@pytest.mark.asyncio
async def test_search_with_special_characters(sec_tools):
    result = await sec_tools.search_10q("TSLA|What was the company's R&D expenditure?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert any(phrase in result.lower() for phrase in ["r&d", "research and development"])

@pytest.mark.asyncio
async def test_invalid_api_key():
    invalid_sec_tools = SECTools(config, "invalid_api_key")
    with pytest.raises(SECToolsError):
        await invalid_sec_tools.search_10q("AAPL|What was the revenue?")

@pytest.mark.asyncio
async def test_embedding_search_content(sec_tools):
    result = await sec_tools.search_10k("NVDA|What is the company's primary business?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert any(word in result.lower() for word in ["gpu", "graphics", "processor"])

@pytest.mark.asyncio
async def test_search_result_length(sec_tools):
    result = await sec_tools.search_10q("ORCL|Summarize the financial performance")
    assert isinstance(result, str)
    assert 100 < len(result) < 10000  # Adjust these bounds as needed

@pytest.mark.asyncio
async def test_concurrent_searches(sec_tools):
    import asyncio
    
    queries = [
        "AAPL|What was the revenue?",
        "MSFT|What were the operating expenses?",
        "GOOGL|What was the net income?",
        "AMZN|What were the total assets?",
        "FB|What was the earnings per share?"
    ]
    
    results = await asyncio.gather(*(sec_tools.search_10q(query) for query in queries))
    
    assert len(results) == len(queries)
    for result in results:
        assert isinstance(result, str)
        assert len(result) > 0