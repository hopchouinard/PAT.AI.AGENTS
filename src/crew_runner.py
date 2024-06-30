from typing import Dict, List, Any
from crewai import Crew, Agent, Task
from logging_config import setup_logging
from exceptions import CrewExecutionError

logger = setup_logging()

def run_crew(agents: Dict[str, Agent], tasks: List[Task], process: str) -> str:
    """Set up and run the crew with given agents and tasks."""
    try:
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=2,
            process=process
        )
        
        logger.info("Starting crew execution")
        result: str = crew.kickoff()
        logger.info("Crew execution completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during crew execution: {e}")
        raise CrewExecutionError(f"Error during crew execution: {e}")

if __name__ == "__main__":
    # Example usage
    try:
        mock_agents = {
            'researcher': Agent(role="Researcher", goal="Research", backstory="Expert researcher"),
            'writer': Agent(role="Writer", goal="Write", backstory="Expert writer")
        }
        mock_tasks = [
            Task(description="Research topic", agent=mock_agents['researcher']),
            Task(description="Write report", agent=mock_agents['writer'])
        ]
        result = run_crew(mock_agents, mock_tasks, "sequential")
        print("Crew execution result:", result)
    except CrewExecutionError as e:
        logger.error(f"Crew execution error: {e}")
        print(f"Crew execution error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An unexpected error occurred. Please check the logs for more information.")