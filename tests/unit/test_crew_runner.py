# tests/unit/test_crew_runner.py

import pytest
from unittest.mock import Mock, patch
from src.crew_runner import CrewRunner
from src.exceptions import CrewExecutionError


@pytest.fixture
def crew_runner():
    mock_config = {"some_config": "value"}
    return CrewRunner(mock_config)


@pytest.mark.asyncio
async def test_run_crew_success(crew_runner):
    mock_agents = {"agent1": Mock(), "agent2": Mock()}
    mock_tasks = [Mock(), Mock()]
    mock_process = "sequential"
    expected_result = "Crew execution result"

    with patch("src.crew_runner.Crew") as MockCrew:
        mock_crew_instance = MockCrew.return_value
        mock_crew_instance.kickoff.return_value = expected_result

        result = await crew_runner.run_crew(mock_agents, mock_tasks, mock_process)

        MockCrew.assert_called_once_with(
            agents=list(mock_agents.values()),
            tasks=mock_tasks,
            verbose=2,
            process=mock_process,
        )
        mock_crew_instance.kickoff.assert_called_once()
        assert result == expected_result


@pytest.mark.asyncio
async def test_run_crew_execution_error(crew_runner):
    mock_agents = {"agent1": Mock()}
    mock_tasks = [Mock()]
    mock_process = "sequential"

    with patch("src.crew_runner.Crew") as MockCrew:
        mock_crew_instance = MockCrew.return_value
        mock_crew_instance.kickoff.side_effect = Exception("Crew execution failed")

        with pytest.raises(CrewExecutionError) as exc_info:
            await crew_runner.run_crew(mock_agents, mock_tasks, mock_process)

        assert (
            str(exc_info.value) == "Error during crew execution: Crew execution failed"
        )


@pytest.mark.asyncio
async def test_run_crew_invalid_process(crew_runner):
    mock_agents = {"agent1": Mock()}
    mock_tasks = [Mock()]
    invalid_process = "invalid_process"

    with patch("src.crew_runner.Crew") as MockCrew:
        await crew_runner.run_crew(mock_agents, mock_tasks, invalid_process)

        MockCrew.assert_called_once_with(
            agents=list(mock_agents.values()),
            tasks=mock_tasks,
            verbose=2,
            process=invalid_process,
        )


@pytest.mark.asyncio
async def test_run_crew_empty_agents(crew_runner):
    mock_agents = {}
    mock_tasks = [Mock()]
    mock_process = "sequential"

    with patch("src.crew_runner.Crew") as MockCrew:
        await crew_runner.run_crew(mock_agents, mock_tasks, mock_process)

        MockCrew.assert_called_once_with(
            agents=[], tasks=mock_tasks, verbose=2, process=mock_process
        )


@pytest.mark.asyncio
async def test_run_crew_empty_tasks(crew_runner):
    mock_agents = {"agent1": Mock()}
    mock_tasks = []
    mock_process = "sequential"

    with patch("src.crew_runner.Crew") as MockCrew:
        await crew_runner.run_crew(mock_agents, mock_tasks, mock_process)

        MockCrew.assert_called_once_with(
            agents=list(mock_agents.values()), tasks=[], verbose=2, process=mock_process
        )
