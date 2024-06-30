from typing import Dict, Any, List
from crewai import Task, Agent
from logging_config import setup_logging
from exceptions import TaskCreationError

logger = setup_logging()

class TaskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def create_tasks(self, crew_config: Dict[str, Any], agents: Dict[str, Agent]) -> List[Task]:
        """Create tasks based on the crew configuration."""
        tasks: List[Task] = []
        
        variable: Dict[str, str] = await self.get_task_variables(crew_config['tasks'][0]['description'])
        
        for task_config in crew_config['tasks']:
            try:
                task_description: str = task_config['description'].format(**variable)
                tasks.append(Task(
                    description=task_description,
                    agent=agents[task_config['agent']],
                    expected_output=task_config['expected_output']
                ))
                logger.info(f"Created task: {task_description[:50]}...")
            except KeyError as e:
                logger.error(f"Missing required configuration for task: {e}")
                raise TaskCreationError(f"Missing required configuration for task: {e}")
            except Exception as e:
                logger.error(f"Failed to create task: {e}")
                raise TaskCreationError(f"Failed to create task: {e}")
        
        return tasks

    async def get_task_variables(self, first_task_description: str) -> Dict[str, str]:
        """Get variables needed for task descriptions."""
        if 'company_name' in first_task_description:
            company_name: str = input("Enter the company name for analysis: ")
            return {'company_name': company_name}
        elif 'ai_prompt' in first_task_description:
            ai_prompt: str = input("Enter the AI prompt for analysis: ")
            return {'ai_prompt': ai_prompt}
        else:
            return {}