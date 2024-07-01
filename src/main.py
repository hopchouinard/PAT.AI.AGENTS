import asyncio
from typing import Dict, List, Any
from logging_config import setup_logging, get_logger, log_execution_time
from config_loader import get_available_crew_configs, load_crew_config
from containers import container
from exceptions import (
    ConfigError,
    APIKeyError,
    FileNotFoundError,
    InvalidConfigError,
    AgentCreationError,
    TaskCreationError,
    CrewExecutionError,
    BaseError,
    AsyncOperationError,
    NetworkError,
    TimeoutError,
    APIError,
)
from crewai import Agent, Task

setup_logging()
logger = get_logger(__name__)


@log_execution_time(logger)
async def get_crew_config() -> Dict[str, Any]:
    crew_files: List[str] = get_available_crew_configs()
    if not crew_files:
        logger.error("No crew configuration files found")
        raise ConfigError("No crew configuration files found.")

    logger.info("Available crew configurations", configurations=crew_files)

    try:
        choice: int = (
            int(input("Enter the number of the configuration you want to run: ")) - 1
        )
        chosen_file: str = crew_files[choice]
        logger.info("Configuration chosen", choice=chosen_file)
    except (ValueError, IndexError):
        logger.error("Invalid configuration choice", exc_info=True)
        raise InvalidConfigError("Invalid configuration choice.")

    return load_crew_config(chosen_file)


async def create_and_run_crew(crew_config: Dict[str, Any]) -> str:
    agent_manager = container.agent_manager()
    task_manager = container.task_manager()
    crew_runner = container.crew_runner()

    agents: Dict[str, Agent] = await agent_manager.create_agents(crew_config)
    logger.info("Agents created", agent_count=len(agents))

    tasks: List[Task] = await task_manager.create_tasks(crew_config, agents)
    logger.info("Tasks created", task_count=len(tasks))

    result: str = await crew_runner.run_crew(
        agents, tasks, crew_config.get("process", container.config.default_crew_process)
    )
    logger.info("Crew execution completed", result_length=len(result))
    return result


async def async_main() -> None:
    try:
        crew_config = await get_crew_config()
        result = await create_and_run_crew(crew_config)
        print("Crew's work result:")
        print(result)
    except AsyncOperationError as e:
        logger.error("Async operation error", error=str(e), exc_info=True)
        print(f"An error occurred during an asynchronous operation: {e}")
    except NetworkError as e:
        logger.error("Network error", error=str(e), exc_info=True)
        print(f"A network error occurred: {e}")
    except TimeoutError as e:
        logger.error("Timeout error", error=str(e), exc_info=True)
        print(f"A timeout occurred: {e}")
    except APIError as e:
        logger.error("API error", error=str(e), exc_info=True)
        print(f"An API error occurred: {e}")
    except (
        ConfigError,
        APIKeyError,
        FileNotFoundError,
        InvalidConfigError,
        AgentCreationError,
        TaskCreationError,
        CrewExecutionError,
        BaseError,
    ) as e:
        logger.error(f"{type(e).__name__}", error=str(e), exc_info=True)
        print(f"{type(e).__name__}: {e}")
    except Exception as e:
        logger.error("Unexpected error", error=str(e), exc_info=True)
        print(
            "An unexpected error occurred. Please check the logs for more information."
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
