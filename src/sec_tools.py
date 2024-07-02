import aiohttp
import asyncio
from typing import Dict, Any, List
from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from sec_api import QueryApi
from unstructured.partition.html import partition_html
from logging_config import setup_logging
from exceptions import SECToolsError, FilingNotFoundError, EmbeddingSearchError
from error_handling import async_retry, with_semaphore, RetryExhaustedError

logger = setup_logging()


class SECTools:
    def __init__(self, config: Dict[str, Any], sec_api_key: str):
        self.config = config
        self.sec_api_key = sec_api_key
        self.semaphore = asyncio.Semaphore(
            self.config.get("max_concurrent_requests", 5)
        )

    @tool("Search 10-Q form")
    async def search_10q(self, query: str) -> str:
        """This method searches for 10-Q forms."""
        logger.debug(f"Searching 10-Q form with query: {query}")
        return await self._search_filing(query, "10-Q")

    @tool("Search 10-K form")
    async def search_10k(self, query: str) -> str:
        """This method searches for 10-K forms."""
        logger.debug(f"Searching 10-K form with query: {query}")
        return await self._search_filing(query, "10-K")

    @async_retry(
        max_retries=3,
        base_delay=1.0,
        exceptions=(aiohttp.ClientError, asyncio.TimeoutError, SECToolsError),
    )
    async def _search_filing(self, query: str, form_type: str) -> str:
        try:
            stock, ask = query.split("|")
        except ValueError:
            logger.error(f"Invalid input format for {form_type} search")
            raise SECToolsError(
                "Invalid input format. Please provide input as 'TICKER|question'."
            )

        try:
            async with aiohttp.ClientSession() as session:
                queryApi = QueryApi(api_key=self.sec_api_key)
                query: Dict[str, Any] = {
                    "query": {
                        "query_string": {
                            "query": f'ticker:{stock} AND formType:"{form_type}"'
                        }
                    },
                    "from": "0",
                    "size": "1",
                    "sort": [{"filedAt": {"order": "desc"}}],
                }

                filings = await with_semaphore(
                    self.semaphore, queryApi.get_filings_async, query, session=session
                )
                if not filings["filings"]:
                    logger.warning(f"No {form_type} filings found for stock: {stock}")
                    raise FilingNotFoundError(
                        f"No {form_type} filings found for stock: {stock}"
                    )

                link: str = filings["filings"][0]["linkToFilingDetails"]
                answer: str = await self.__embedding_search(link, ask)
                logger.info(f"{form_type} search completed for stock: {stock}")
                return answer
        except FilingNotFoundError:
            return f"Sorry, I couldn't find any {form_type} filing for this stock. Please check if the ticker is correct."
        except RetryExhaustedError as e:
            logger.error(f"Retry attempts exhausted for {form_type} search: {e}")
            raise SECToolsError(
                f"Failed to complete {form_type} search after multiple attempts. Please try again later."
            )
        except Exception as e:
            logger.error(f"Error in {form_type} search: {e}")
            raise SECToolsError(f"Error in {form_type} search: {str(e)}")

    @async_retry(
        max_retries=3,
        base_delay=1.0,
        exceptions=(aiohttp.ClientError, asyncio.TimeoutError, EmbeddingSearchError),
    )
    async def __embedding_search(self, url: str, ask: str) -> str:
        logger.debug(f"Performing embedding search for URL: {url}")
        try:
            text: str = await self.__download_form_html(url)
            elements: List[Any] = partition_html(text=text)
            content: str = "\n".join([str(el) for el in elements])
            text_splitter: CharacterTextSplitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=self.config["embedding_chunk_size"],
                chunk_overlap=self.config["embedding_chunk_overlap"],
                length_function=len,
                is_separator_regex=False,
            )
            docs: List[Any] = text_splitter.create_documents([content])

            embeddings: OllamaEmbeddings = OllamaEmbeddings(
                model=self.config["embedding_model"]
            )

            retriever: Any = FAISS.from_documents(docs, embeddings).as_retriever()

            answers: List[Any] = await retriever.aget_relevant_documents(ask, top_k=4)
            answer: str = "\n\n".join([a.page_content for a in answers])
            logger.debug("Embedding search completed")
            return answer
        except Exception as e:
            logger.error(f"Error in embedding search: {e}")
            raise EmbeddingSearchError(f"Error in embedding search: {str(e)}")

    @async_retry(
        max_retries=3,
        base_delay=1.0,
        exceptions=(aiohttp.ClientError, asyncio.TimeoutError),
    )
    async def __download_form_html(self, url: str) -> str:
        logger.debug(f"Downloading HTML from URL: {url}")
        headers: Dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                response.raise_for_status()
                logger.debug(f"HTML download completed, status code: {response.status}")
                return await response.text()
