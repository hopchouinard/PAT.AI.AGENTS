from typing import Dict, Any, Union
from langchain.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from logging_config import setup_logging
from exceptions import (
    ConfigError, APIKeyError, SearchToolError,
    NetworkError, TimeoutError, APIError,
    SearchQueryError, SearchResultParseError
)
import aiohttp
import asyncio
import json

logger = setup_logging()

class SearchTool:
    def __init__(self, config: Dict[str, Any], serper_api_key: str):
        self.config = config
        self.serper_api_key = serper_api_key

    async def create_search_tool(self) -> Tool:
        async def search_function(*args: Any, **kwargs: Any) -> str:
            logger.debug(f"Search function called with args: {args}, kwargs: {kwargs}")
            
            query: Union[str, Dict[str, Any]] = args[0] if args else kwargs.get('query', '')
            
            logger.debug(f"Extracted query: {query}")
            
            try:
                if isinstance(query, dict):
                    actual_query: str = query.get('query', '')
                elif isinstance(query, str):
                    actual_query = query
                else:
                    actual_query = str(query)
                
                logger.debug(f"Processed query: {actual_query}")
                
                result: str = await self.async_search(actual_query)
                logger.debug(f"Search result: {result[:self.config['search_result_limit']]}...")
                return result
            except SearchQueryError as e:
                logger.error(f"Error processing search query: {e}")
                return f"Error processing search query: {str(e)}"
            except Exception as e:
                logger.error(f"Unexpected error in search function: {e}")
                raise SearchToolError(f"An unexpected error occurred while processing the search query: {str(e)}")
        
        return Tool(
            name="Search",
            func=search_function,
            description="Search the internet for current information. Input should be a string containing the search query."
        )

    async def async_search(self, query: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": self.serper_api_key},
                    params={"q": query},
                    timeout=30
                ) as response:
                    response.raise_for_status()
                    search_results = await response.json()
                    
                    # Process the search results as needed
                    return self.process_search_results(search_results)
            except aiohttp.ClientError as e:
                logger.error(f"Network error during search: {e}")
                raise NetworkError(f"Network error during search: {str(e)}")
            except asyncio.TimeoutError:
                logger.error("Search request timed out")
                raise TimeoutError("Search request timed out")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding search results: {e}")
                raise SearchResultParseError(f"Error decoding search results: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during search: {e}")
                raise APIError(f"Unexpected error during search: {str(e)}")

    def process_search_results(self, results: Dict[str, Any]) -> str:
        try:
            # Implement your logic to process and format search results
            # This is a simple example, you might want to format it differently
            formatted_results = json.dumps(results, indent=2)
            return formatted_results
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            raise SearchResultParseError(f"Error processing search results: {str(e)}")

async def create_search_tool(config: Dict[str, Any], serper_api_key: str) -> Tool:
    search_tool = SearchTool(config, serper_api_key)
    return await search_tool.create_search_tool()