from typing import Dict, Any, List
from crewai import Task, Agent
from logging_config import setup_logging
from exceptions import TaskCreationError

logger = setup_logging()

def create_tasks(crew_config: Dict[str, Any], agents: Dict[str, Agent]) -> List[Task]:
    """Create tasks based on the crew configuration."""
    tasks: List[Task] = []
    
    # Get input for the crew
    variable: Dict[str, str] = get_task_variables(crew_config['tasks'][0]['description'])
    
    # Create tasks
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

def get_task_variables(first_task_description: str) -> Dict[str, str]:
    """Get variables needed for task descriptions."""
    if 'company_name' in first_task_description:
        company_name: str = input("Enter the company name for analysis: ")
        return {'company_name': company_name}
    elif 'ai_prompt' in first_task_description:
        ai_prompt: str = input("Enter the AI prompt for analysis: ")
        return {'ai_prompt': ai_prompt}
    else:
        return {}

if __name__ == "__main__":
    # Example usage
    try:
        mock_config = {
            'tasks': [
                {'description': 'Analyze {company_name}', 'agent': 'analyst', 'expected_output': 'Analysis'}
            ]
        }
        mock_agents = {'analyst': Agent(role="Analyst", goal="Analyze", backstory="Expert analyst")}
        tasks = create_tasks(mock_config, mock_agents)
        print(f"Created {len(tasks)} tasks")
    except TaskCreationError as e:
        logger.error(f"Task creation error: {e}")
        print(f"Task creation error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An unexpected error occurred. Please check the logs for more information.")