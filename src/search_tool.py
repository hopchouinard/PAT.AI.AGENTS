import json
import logging
import os
from langchain.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_search_tool():
    # Check if the Serper API key is set
    serper_api_key = os.getenv('SERPER_API_KEY')
    if not serper_api_key:
        logger.error("SERPER_API_KEY is not set in the environment variables")
        raise ValueError("SERPER_API_KEY is not set")

    # Initialize GoogleSerperAPIWrapper with the API key
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
    
    def search_function(*args, **kwargs):
        logger.debug(f"Search function called with args: {args}, kwargs: {kwargs}")
        
        # Extract the query from args or kwargs
        if args:
            query = args[0]
        elif kwargs:
            query = kwargs.get('query', '')
        else:
            query = ''
        
        logger.debug(f"Extracted query: {query}")
        
        # Handle different input types
        try:
            if isinstance(query, dict):
                actual_query = query.get('query', '')
            elif isinstance(query, str):
                # Always treat string input as the query itself
                actual_query = query
            else:
                actual_query = str(query)
            
            logger.debug(f"Processed query: {actual_query}")
            
            result = search.run(actual_query)
            logger.debug(f"Search result: {result[:100]}...")  # Log first 100 chars of result
            return result
        except Exception as e:
            logger.error(f"Error in search function: {e}")
            return f"An error occurred while processing the search query: {str(e)}"
    
    return Tool(
        name="Search",
        func=search_function,
        description="Search the internet for current information. Input should be a string containing the search query."
    )