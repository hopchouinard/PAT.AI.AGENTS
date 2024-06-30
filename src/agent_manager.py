from typing import Dict, Any, List
from crewai import Agent
from langchain_community.llms import Ollama
from logging_config import setup_logging
from exceptions import AgentCreationError

logger = setup_logging()

class AgentManager:
    def __init__(self, config: Dict[str, Any], ollama_llm: Ollama, search_tool: Any, sec_tools: Any):
        self.config = config
        self.ollama_llm = ollama_llm
        self.search_tool = search_tool
        self.sec_tools = sec_tools

    async def create_agents(self, crew_config: Dict[str, Any]) -> Dict[str, Agent]:
        """Create agents based on the crew configuration."""
        agents: Dict[str, Agent] = {}

        for agent_name, agent_config in crew_config['agents'].items():
            tools: List[Any] = []
            if agent_config.get('use_search_tool', False):
                tools.append(await self.search_tool)
            if agent_config.get('use_sec_tools', False):
                tools.extend([self.sec_tools.search_10q, self.sec_tools.search_10k])
            
            try:
                agents[agent_name] = Agent(
                    role=agent_config['role'],
                    goal=agent_config['goal'],
                    backstory=agent_config['backstory'],
                    verbose=agent_config.get('verbose', True),
                    allow_delegation=agent_config.get('allow_delegation', False),
                    tools=tools,
                    llm=self.ollama_llm
                )
                logger.info(f"Created agent: {agent_name}")
            except KeyError as e:
                logger.error(f"Missing required configuration for agent {agent_name}: {e}")
                raise AgentCreationError(f"Missing required configuration for agent {agent_name}: {e}")
            except Exception as e:
                logger.error(f"Failed to create agent {agent_name}: {e}")
                raise AgentCreationError(f"Failed to create agent {agent_name}: {e}")
        
        return agents