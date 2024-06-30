class BaseError(Exception):
    """Base class for custom exceptions."""
    pass

# Configuration related exceptions
class ConfigError(BaseError):
    """Base exception for configuration related errors."""
    pass

class APIKeyError(ConfigError):
    """Raised when there's an issue with API keys."""
    pass

class FileNotFoundError(ConfigError):
    """Raised when a required file is not found."""
    pass

class InvalidConfigError(ConfigError):
    """Raised when the configuration is invalid."""
    pass

# Agent related exceptions
class AgentError(BaseError):
    """Base exception for agent related errors."""
    pass

class AgentCreationError(AgentError):
    """Raised when there's an error creating an agent."""
    pass

# Task related exceptions
class TaskError(BaseError):
    """Base exception for task related errors."""
    pass

class TaskCreationError(TaskError):
    """Raised when there's an error creating a task."""
    pass

# Crew related exceptions
class CrewError(BaseError):
    """Base exception for crew related errors."""
    pass

class CrewExecutionError(CrewError):
    """Raised when there's an error during crew execution."""
    pass

# SEC Tools related exceptions
class SECToolsError(BaseError):
    """Base exception for SEC tools related errors."""
    pass

class FilingNotFoundError(SECToolsError):
    """Raised when a filing is not found."""
    pass

class EmbeddingSearchError(SECToolsError):
    """Raised when there's an error during embedding search."""
    pass

# Search tool related exceptions
class SearchToolError(BaseError):
    """Base exception for search tool related errors."""
    pass

class SearchQueryError(SearchToolError):
    """Raised when there's an error processing a search query."""
    pass

# Async-specific exceptions
class AsyncOperationError(BaseError):
    """Base exception for asynchronous operation errors."""
    pass

class NetworkError(AsyncOperationError):
    """Raised when there's a network-related error in async operations."""
    pass

class TimeoutError(AsyncOperationError):
    """Raised when an async operation times out."""
    pass

class APIError(AsyncOperationError):
    """Raised when there's an error related to API calls."""
    pass

class ParseError(AsyncOperationError):
    """Raised when there's an error parsing async operation results."""
    pass

# SEC Tools specific async exceptions
class SECFilingDownloadError(AsyncOperationError):
    """Raised when there's an error downloading SEC filings."""
    pass

class SECFilingParseError(AsyncOperationError):
    """Raised when there's an error parsing SEC filings."""
    pass

# Search tool specific async exceptions
class SearchQueryError(AsyncOperationError):
    """Raised when there's an error processing a search query."""
    pass

class SearchResultParseError(AsyncOperationError):
    """Raised when there's an error parsing search results."""
    pass