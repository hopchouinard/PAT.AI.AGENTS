from typing import Dict, Any
from langchain_community.llms import Ollama
from search_tool import create_search_tool
from sec_tools import SECTools
from config_loader import load_main_config
from exceptions import ConfigError, OllamaInitializationError

class Dependencies:
    def __init__(self):
        self.config: Dict[str, Any] = self._load_config()
        self.ollama_llm: Ollama = self._initialize_ollama()
        self.search_tool = create_search_tool()
        self.sec_tools = [SECTools.search_10q, SECTools.search_10k]

    def _load_config(self) -> Dict[str, Any]:
        try:
            return load_main_config()
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {str(e)}")

    def _initialize_ollama(self) -> Ollama:
        try:
            model_name = self.config['default_llm_model']
            return Ollama(model=model_name)
        except Exception as e:
            raise OllamaInitializationError(f"Failed to initialize Ollama: {str(e)}")

dependencies = Dependencies()