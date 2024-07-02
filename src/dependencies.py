from typing import Dict, Any
from langchain_community.llms import Ollama
from search_tool import create_search_tool
from sec_tools import SECTools
from config import config
from exceptions import ConfigError, OllamaInitializationError
from agent_manager import AgentManager
from task_manager import TaskManager
from crew_runner import CrewRunner
from embedding_manager import EmbeddingManager

class Dependencies:
    def __init__(self):
        self.config: Dict[str, Any] = config.dict()
        self.ollama_llm: Ollama = self._initialize_ollama()
        self.search_tool = create_search_tool(self.config, config.serper_api_key.get_secret_value())
        self.sec_tools = SECTools(self.config, config.sec_api_key.get_secret_value())
        self.embedding_manager = EmbeddingManager()

        self.agent_manager = AgentManager(
            self.config,
            self.ollama_llm,
            self.search_tool,
            self.sec_tools,
            self.embedding_manager
        )
        self.task_manager = TaskManager(self.config)
        self.crew_runner = CrewRunner(self.config)

    def _initialize_ollama(self) -> Ollama:
        try:
            return Ollama(model=self.config["default_llm_model"])
        except Exception as e:
            raise OllamaInitializationError(f"Failed to initialize Ollama: {str(e)}")

dependencies = Dependencies()