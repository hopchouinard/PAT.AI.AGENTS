import os
from typing import Dict, Any, Tuple, List
from utils import load_yaml_config, load_environment_variables, get_crew_configs
from logging_config import setup_logging
from exceptions import ConfigError, FileNotFoundError, InvalidConfigError, APIKeyError

logger = setup_logging()

def load_main_config() -> Dict[str, Any]:
    """Load the main configuration file."""
    try:
        config = load_yaml_config('config.yaml')
        validate_config(config)
        return config
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except InvalidConfigError as e:
        logger.error(f"Invalid configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading main configuration: {e}")
        raise ConfigError(f"Failed to load main configuration: {e}")

def load_crew_config(crew_file: str) -> Dict[str, Any]:
    """Load a specific crew configuration file."""
    try:
        config = load_yaml_config(os.path.join('crew', crew_file))
        validate_crew_config(config)
        return config
    except FileNotFoundError as e:
        logger.error(f"Crew configuration file not found: {e}")
        raise
    except InvalidConfigError as e:
        logger.error(f"Invalid crew configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading crew configuration: {e}")
        raise ConfigError(f"Failed to load crew configuration {crew_file}: {e}")

def get_available_crew_configs() -> List[str]:
    """Get a list of available crew configuration files."""
    try:
        return get_crew_configs()
    except Exception as e:
        logger.error(f"Failed to get crew configurations: {e}")
        raise ConfigError(f"Failed to get crew configurations: {e}")

def setup_environment(config: Dict[str, Any]) -> Tuple[str, str]:
    """Set up environment variables based on the configuration."""
    try:
        sec_api_key, serper_api_key = load_environment_variables(config)
        if not sec_api_key:
            raise APIKeyError("SEC_API_KEY is missing")
        if not serper_api_key:
            raise APIKeyError("SERPER_API_KEY is missing")
        os.environ['SERPER_API_KEY'] = serper_api_key
        return sec_api_key, serper_api_key
    except APIKeyError as e:
        logger.error(f"API key error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error setting up environment: {e}")
        raise ConfigError(f"Failed to set up environment: {e}")

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the main configuration."""
    required_keys = ['default_llm_model', 'log_level', 'embedding_chunk_size', 'embedding_chunk_overlap']
    for key in required_keys:
        if key not in config:
            raise InvalidConfigError(f"Missing required configuration key: {key}")
    
    if not isinstance(config.get('embedding_chunk_size'), int) or config.get('embedding_chunk_size', 0) <= 0:
        raise InvalidConfigError("embedding_chunk_size must be a positive integer")

    if not isinstance(config.get('embedding_chunk_overlap'), int) or config.get('embedding_chunk_overlap', 0) < 0:
        raise InvalidConfigError("embedding_chunk_overlap must be a non-negative integer")

def validate_crew_config(config: Dict[str, Any]) -> None:
    """Validate the crew configuration."""
    if 'agents' not in config or not isinstance(config['agents'], dict):
        raise InvalidConfigError("Crew configuration must contain an 'agents' dictionary")
    if 'tasks' not in config or not isinstance(config['tasks'], list):
        raise InvalidConfigError("Crew configuration must contain a 'tasks' list")
    # Add more specific validations as needed