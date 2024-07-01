# tests/unit/test_task_manager.py

import pytest
from unittest.mock import Mock, patch
from src.task_manager import TaskManager
from src.exceptions import TaskCreationError


@pytest.fixture
def task_manager():
    mock_config = {"some_config": "value"}
    return TaskManager(mock_config)


@pytest.mark.asyncio
async def test_create_tasks_success(task_manager):
    mock_crew_config = {
        "tasks": [
            {
                "description": "Analyze {company_name}",
                "agent": "analyst",
                "expected_output": "Analysis report",
            },
            {
                "description": "Research {company_name} competitors",
                "agent": "researcher",
                "expected_output": "Competitor list",
            },
        ]
    }
    mock_agents = {"analyst": Mock(), "researcher": Mock()}

    with patch("builtins.input", return_value="Test Company"):
        tasks = await task_manager.create_tasks(mock_crew_config, mock_agents)

    assert len(tasks) == 2
    assert tasks[0].description == "Analyze Test Company"
    assert tasks[0].agent == mock_agents["analyst"]
    assert tasks[0].expected_output == "Analysis report"
    assert tasks[1].description == "Research Test Company competitors"
    assert tasks[1].agent == mock_agents["researcher"]
    assert tasks[1].expected_output == "Competitor list"


@pytest.mark.asyncio
async def test_create_tasks_missing_required_field(task_manager):
    mock_crew_config = {
        "tasks": [
            {
                "description": "Analyze {company_name}",
                # 'agent' is missing
                "expected_output": "Analysis report",
            }
        ]
    }
    mock_agents = {"analyst": Mock()}

    with pytest.raises(TaskCreationError) as exc_info:
        await task_manager.create_tasks(mock_crew_config, mock_agents)

    assert "Missing required configuration for task" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_tasks_empty_config(task_manager):
    mock_crew_config = {"tasks": []}
    mock_agents = {}

    tasks = await task_manager.create_tasks(mock_crew_config, mock_agents)
    assert len(tasks) == 0


@pytest.mark.asyncio
async def test_create_tasks_missing_agent(task_manager):
    mock_crew_config = {
        "tasks": [
            {
                "description": "Analyze {company_name}",
                "agent": "non_existent_agent",
                "expected_output": "Analysis report",
            }
        ]
    }
    mock_agents = {"analyst": Mock()}

    with pytest.raises(TaskCreationError) as exc_info:
        await task_manager.create_tasks(mock_crew_config, mock_agents)

    assert "Agent non_existent_agent not found for task" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_task_variables_company_name(task_manager):
    with patch("builtins.input", return_value="Test Company"):
        variables = await task_manager.get_task_variables("Analyze {company_name}")

    assert variables == {"company_name": "Test Company"}


@pytest.mark.asyncio
async def test_get_task_variables_ai_prompt(task_manager):
    with patch("builtins.input", return_value="Generate a report"):
        variables = await task_manager.get_task_variables("Execute {ai_prompt}")

    assert variables == {"ai_prompt": "Generate a report"}


@pytest.mark.asyncio
async def test_get_task_variables_no_variable(task_manager):
    variables = await task_manager.get_task_variables("Perform general analysis")

    assert variables == {}


@pytest.mark.asyncio
async def test_create_tasks_with_multiple_variables(task_manager):
    mock_crew_config = {
        "tasks": [
            {
                "description": "Analyze {company_name} in {industry}",
                "agent": "analyst",
                "expected_output": "Industry analysis",
            }
        ]
    }
    mock_agents = {"analyst": Mock()}

    with patch(
        "src.task_manager.TaskManager.get_task_variables",
        return_value={"company_name": "Test Company", "industry": "Tech"},
    ):
        tasks = await task_manager.create_tasks(mock_crew_config, mock_agents)

    assert len(tasks) == 1
    assert tasks[0].description == "Analyze Test Company in Tech"
