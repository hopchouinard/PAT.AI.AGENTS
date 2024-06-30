from dependency_injector import containers, providers
from langchain_community.llms import Ollama
from config import config
from sec_tools import SECTools
from search_tool import SearchTool, create_search_tool
from agent_manager import AgentManager
from task_manager import TaskManager
from crew_runner import CrewRunner

class Container(containers.DeclarativeContainer):
    # Use the pydantic config
    config = providers.Configuration(pydantic_settings=[config])

    # Core components
    ollama_llm = providers.Factory(
        Ollama,
        model=config.default_llm_model
    )

    sec_tools = providers.Factory(
        SECTools,
        config=config,
        sec_api_key=config.sec_api_key.get_secret_value()
    )

    search_tool = providers.Factory(
        create_search_tool,
        config=config,
        serper_api_key=config.serper_api_key.get_secret_value()
    )

    # Managers
    agent_manager = providers.Factory(
        AgentManager,
        config=config,
        ollama_llm=ollama_llm,
        search_tool=search_tool,
        sec_tools=sec_tools
    )

    task_manager = providers.Factory(
        TaskManager,
        config=config
    )

    crew_runner = providers.Factory(
        CrewRunner,
        config=config
    )

# Create and configure the container
container = Container()