import asyncio
from typing import Dict, List, Any
from logging_config import setup_logging
from config_loader import load_crew_config, get_available_crew_configs, setup_environment
from containers import container
from exceptions import (
    ConfigError, APIKeyError, FileNotFoundError, InvalidConfigError,
    AgentCreationError, TaskCreationError, CrewExecutionError, BaseError,
    AsyncOperationError, NetworkError, TimeoutError, APIError
)

logger = setup_logging()

async def async_main() -> None:
    try:
        # Set up environment
        setup_environment(container.config())

        # Get available crew configurations
        crew_files: List[str] = get_available_crew_configs()
        
        if not crew_files:
            raise ConfigError("No crew configuration files found.")
        
        print("Available crew configurations:")
        for i, file in enumerate(crew_files, 1):
            print(f"{i}. {file}")
        
        # Let the user choose a configuration
        try:
            choice: int = int(input("Enter the number of the configuration you want to run: ")) - 1
            chosen_file: str = crew_files[choice]
        except (ValueError, IndexError):
            raise InvalidConfigError("Invalid configuration choice.")
        
        # Load the chosen crew configuration
        crew_config: Dict[str, Any] = load_crew_config(chosen_file)
        
        # Get dependencies from the container
        agent_manager = container.agent_manager()
        task_manager = container.task_manager()
        crew_runner = container.crew_runner()
        
        try:
            # Create agents and tasks
            agents: Dict[str, Any] = await agent_manager.create_agents(crew_config)
            tasks: List[Any] = await task_manager.create_tasks(crew_config, agents)
            
            # Run the crew
            result: str = await crew_runner.run_crew(agents, tasks, crew_config.get('process', container.config.default_crew_process()))
            
            print("Crew's work result:")
            print(result)
        except AsyncOperationError as e:
            logger.error(f"Async operation error: {e}")
            print(f"An error occurred during an asynchronous operation: {e}")
        except NetworkError as e:
            logger.error(f"Network error: {e}")
            print(f"A network error occurred: {e}")
        except TimeoutError as e:
            logger.error(f"Timeout error: {e}")
            print(f"A timeout occurred: {e}")
        except APIError as e:
            logger.error(f"API error: {e}")
            print(f"An API error occurred: {e}")

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
    except APIKeyError as e:
        logger.error(f"API key error: {e}")
        print(f"API key error: {e}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"File not found: {e}")
    except InvalidConfigError as e:
        logger.error(f"Invalid configuration: {e}")
        print(f"Invalid configuration: {e}")
    except AgentCreationError as e:
        logger.error(f"Agent creation error: {e}")
        print(f"Agent creation error: {e}")
    except TaskCreationError as e:
        logger.error(f"Task creation error: {e}")
        print(f"Task creation error: {e}")
    except CrewExecutionError as e:
        logger.error(f"Crew execution error: {e}")
        print(f"Crew execution error: {e}")
    except BaseError as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred. Please check the logs for more information.")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()