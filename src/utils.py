import os
import yaml
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv
from logging_config import setup_logging
from exceptions import ConfigError, FileNotFoundError, APIKeyError

# Set up logging
logger = setup_logging()


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file is not found.
        ConfigError: If there's an error parsing the YAML file.
    """
    try:
        with open(file_path, "r") as file:
            config: Dict[str, Any] = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {str(e)}")
        raise ConfigError(f"Error parsing YAML file {file_path}: {str(e)}")


def load_environment_variables(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Load environment variables from a .env file and validate required variables.

    Args:
        config (dict): Configuration dictionary containing default values.

    Returns:
        tuple: A tuple containing the SEC API key and Serper API key.

    Raises:
        FileNotFoundError: If the .env file is not found.
        APIKeyError: If required environment variables are missing.
    """
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    project_root: str = os.path.dirname(current_dir)
    env_path: str = os.path.join(project_root, ".env")

    if not os.path.exists(env_path):
        logger.error(f".env file not found at {env_path}")
        raise FileNotFoundError(f".env file not found at {env_path}")

    load_dotenv(env_path)

    sec_api_key: str = os.getenv("SEC_API_KEY", config.get("sec_api_key", ""))
    serper_api_key: str = os.getenv("SERPER_API_KEY", config.get("serper_api_key", ""))

    if not sec_api_key:
        logger.error("SEC_API_KEY is missing in both .env file and config")
        raise APIKeyError("SEC_API_KEY is missing")
    if not serper_api_key:
        logger.error("SERPER_API_KEY is missing in both .env file and config")
        raise APIKeyError("SERPER_API_KEY is missing")

    logger.info("Environment variables loaded successfully")
    return sec_api_key, serper_api_key


def get_project_root() -> str:
    """
    Get the root directory of the project.

    Returns:
        str: Absolute path to the project root directory.
    """
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def get_crew_configs() -> List[str]:
    """
    Get a list of available crew configuration files.

    Returns:
        list: A list of YAML file names in the 'crew' folder.
    """
    crew_folder: str = os.path.join(get_project_root(), "crew")
    return [f for f in os.listdir(crew_folder) if f.endswith(".yaml")]


if __name__ == "__main__":
    try:
        config: Dict[str, Any] = load_yaml_config("config.yaml")
        sec_api_key, serper_api_key = load_environment_variables(config)
        print(
            f"SEC API Key: {sec_api_key[:5]}..."
        )  # Print first 5 characters for security
        print(f"Serper API Key: {serper_api_key[:5]}...")
        print(f"Project Root: {get_project_root()}")
        print(f"Available Crew Configs: {get_crew_configs()}")
    except (FileNotFoundError, ConfigError, APIKeyError) as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(
            "An unexpected error occurred. Please check the logs for more information."
        )
