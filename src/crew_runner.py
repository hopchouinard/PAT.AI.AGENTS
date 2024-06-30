from typing import Dict, List, Any
from crewai import Crew, Agent, Task
from logging_config import setup_logging
from exceptions import CrewExecutionError

logger = setup_logging()

class CrewRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def run_crew(self, agents: Dict[str, Agent], tasks: List[Task], process: str) -> str:
        """Set up and run the crew with given agents and tasks asynchronously."""
        try:
            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                verbose=2,
                process=process
            )
            
            logger.info("Starting crew execution")
            result: str = await crew.kickoff()
            logger.info("Crew execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error during crew execution: {e}")
            raise CrewExecutionError(f"Error during crew execution: {e}")