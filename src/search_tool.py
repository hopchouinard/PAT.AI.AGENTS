from typing import Dict, Any, Union
from langchain.tools import Tool
from logging_config import setup_logging
from exceptions import SearchToolError
from error_handling import async_retry, RetryExhaustedError
import aiohttp
import asyncio
import json

logger = setup_logging()


class SearchTool:
    def __init__(self, config: Dict[str, Any], serper_api_key: str):
        self.config = config
        self.serper_api_key = serper_api_key

    async def create_search_tool(self) -> Tool:
        @async_retry(
            max_retries=3,
            base_delay=1.0,
            exceptions=(aiohttp.ClientError, asyncio.TimeoutError, SearchToolError),
        )
        async def search_function(*args: Any, **kwargs: Any) -> str:
            logger.debug(f"Search function called with args: {args}, kwargs: {kwargs}")

            query: Union[str, Dict[str, Any]] = (
                args[0] if args else kwargs.get("query", "")
            )

            logger.debug(f"Extracted query: {query}")

            try:
                if isinstance(query, dict):
                    actual_query: str = query.get("query", "")
                elif isinstance(query, str):
                    actual_query = query
                else:
                    actual_query = str(query)

                logger.debug(f"Processed query: {actual_query}")

                result: str = await self.async_search(actual_query)
                logger.debug(
                    f"Search result: {result[:self.config['search_result_limit']]}..."
                )
                return result
            except RetryExhaustedError as e:
                logger.error(f"Retry attempts exhausted for search query: {e}")
                raise SearchToolError(
                    "Failed to complete search after multiple attempts. Please try again later."
                )
            except Exception as e:
                logger.error(f"Error in search function: {e}")
                raise SearchToolError(
                    f"An error occurred while processing the search query: {str(e)}"
                )

        return Tool(
            name="Search",
            func=search_function,
            description="Search the internet for current information. Input should be a string containing the search query.",
        )

    @async_retry(
        max_retries=3,
        base_delay=1.0,
        exceptions=(aiohttp.ClientError, asyncio.TimeoutError),
    )
    async def async_search(self, query: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.serper_api_key},
                params={"q": query},
                timeout=30,
            ) as response:
                response.raise_for_status()
                search_results = await response.json()
                return self.process_search_results(search_results)

    def process_search_results(self, results: Dict[str, Any]) -> str:
        try:
            # Implement your logic to process and format search results
            formatted_results = json.dumps(results, indent=2)
            return formatted_results
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            raise SearchToolError(f"Error processing search results: {str(e)}")


async def create_search_tool(config: Dict[str, Any], serper_api_key: str) -> Tool:
    search_tool = SearchTool(config, serper_api_key)
    return await search_tool.create_search_tool()
