from typing import Dict, Any, List
from crewai import Agent
from langchain_community.llms import Ollama
from logging_config import LoggerMixin, log_execution_time
from exceptions import AgentCreationError
from embedding_manager import EmbeddingManager

class AgentManager(LoggerMixin):
    def __init__(
        self,
        config: Dict[str, Any],
        ollama_llm: Ollama,
        search_tool: Any,
        sec_tools: Any,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self.config: Dict[str, Any] = config
        self.ollama_llm: Ollama = ollama_llm
        self.search_tool: Any = search_tool
        self.sec_tools: Any = sec_tools
        self.embedding_manager: EmbeddingManager = embedding_manager

    @log_execution_time(logger=None)
    async def create_agents(self, crew_config: Dict[str, Any]) -> Dict[str, Agent]:
        """Create agents based on the crew configuration."""
        agents: Dict[str, Agent] = {}

        for agent_name, agent_config in crew_config["agents"].items():
            tools: List[Any] = []
            if agent_config.get("use_search_tool", False):
                tools.append(await self.search_tool)
            if agent_config.get("use_sec_tools", False):
                tools.extend([self.sec_tools.search_10q, self.sec_tools.search_10k])

            try:
                agents[agent_name] = Agent(
                    role=agent_config["role"],
                    goal=agent_config["goal"],
                    backstory=agent_config["backstory"],
                    verbose=agent_config.get("verbose", True),
                    allow_delegation=agent_config.get("allow_delegation", False),
                    tools=tools,
                    llm=self.ollama_llm,
                    # Use a custom memory implementation if needed
                    # memory=CustomMemory(self.embedding_manager),
                )
                self.logger.info("Agent created", agent_name=agent_name)
            except KeyError as e:
                self.logger.error(
                    "Missing required configuration for agent",
                    agent_name=agent_name,
                    error=str(e),
                )
                raise AgentCreationError(
                    f"Missing required configuration for agent {agent_name}: {e}"
                )
            except Exception as e:
                self.logger.error(
                    "Failed to create agent", agent_name=agent_name, error=str(e)
                )
                raise AgentCreationError(f"Failed to create agent {agent_name}: {e}")

        self.logger.info("All agents created", agent_count=len(agents))
        return agents