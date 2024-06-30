# tests/e2e/test_main.py

import pytest
from unittest.mock import patch, MagicMock
import asyncio
import yaml
from src.main import async_main
from src.config import config
from src.exceptions import ConfigError, APIKeyError, CrewExecutionError

@pytest.fixture
def mock_dependencies():
    with patch('src.main.container') as mock_container:
        mock_agent_manager = MagicMock()
        mock_task_manager = MagicMock()
        mock_crew_runner = MagicMock()
        
        mock_container.agent_manager.return_value = mock_agent_manager
        mock_container.task_manager.return_value = mock_task_manager
        mock_container.crew_runner.return_value = mock_crew_runner
        
        yield mock_agent_manager, mock_task_manager, mock_crew_runner

@pytest.mark.asyncio
async def test_main_successful_execution(mock_dependencies):
    mock_agent_manager, mock_task_manager, mock_crew_runner = mock_dependencies
    
    # Mock user input
    with patch('builtins.input', return_value='1'):
        # Mock configuration loading
        with patch('src.main.get_available_crew_configs', return_value=['test_config.yaml']):
            with patch('src.main.load_crew_config', return_value={
                'agents': {'test_agent': {}},
                'tasks': [{'description': 'Test task', 'agent': 'test_agent'}],
                'process': 'sequential'
            }):
                # Mock agent and task creation
                mock_agent_manager.create_agents.return_value = {'test_agent': MagicMock()}
                mock_task_manager.create_tasks.return_value = [MagicMock()]
                
                # Mock crew execution
                mock_crew_runner.run_crew.return_value = "Test execution result"
                
                # Run the main function
                await async_main()
                
                # Assertions
                mock_agent_manager.create_agents.assert_called_once()
                mock_task_manager.create_tasks.assert_called_once()
                mock_crew_runner.run_crew.assert_called_once()

@pytest.mark.asyncio
async def test_main_no_crew_configs():
    with patch('src.main.get_available_crew_configs', return_value=[]):
        with pytest.raises(ConfigError, match="No crew configuration files found"):
            await async_main()

@pytest.mark.asyncio
async def test_main_invalid_crew_choice(mock_dependencies):
    with patch('builtins.input', side_effect=['3', '1']):  # First input invalid, second valid
        with patch('src.main.get_available_crew_configs', return_value=['config1.yaml', 'config2.yaml']):
            with patch('src.main.load_crew_config', return_value={}):
                with patch('src.main.setup_environment'):
                    await async_main()
                    # Assert that the function recovered from the invalid input

@pytest.mark.asyncio
async def test_main_api_key_error(mock_dependencies):
    with patch('src.main.setup_environment', side_effect=APIKeyError("Missing API key")):
        with pytest.raises(SystemExit):
            await async_main()

@pytest.mark.asyncio
async def test_main_crew_execution_error(mock_dependencies):
    mock_agent_manager, mock_task_manager, mock_crew_runner = mock_dependencies
    
    with patch('builtins.input', return_value='1'):
        with patch('src.main.get_available_crew_configs', return_value=['test_config.yaml']):
            with patch('src.main.load_crew_config', return_value={}):
                mock_agent_manager.create_agents.return_value = {'test_agent': MagicMock()}
                mock_task_manager.create_tasks.return_value = [MagicMock()]
                mock_crew_runner.run_crew.side_effect = CrewExecutionError("Execution failed")
                
                with pytest.raises(SystemExit):
                    await async_main()

@pytest.mark.asyncio
async def test_main_unexpected_error(mock_dependencies):
    mock_agent_manager, mock_task_manager, mock_crew_runner = mock_dependencies
    
    with patch('builtins.input', return_value='1'):
        with patch('src.main.get_available_crew_configs', return_value=['test_config.yaml']):
            with patch('src.main.load_crew_config', side_effect=Exception("Unexpected error")):
                with pytest.raises(SystemExit):
                    await async_main()

@pytest.mark.asyncio
async def test_main_with_real_config(mock_dependencies):
    mock_agent_manager, mock_task_manager, mock_crew_runner = mock_dependencies
    
    # Use a real config file from your project
    real_config_path = './src/config.yaml'
    
    with patch('builtins.input', return_value='1'):
        with patch('src.main.get_available_crew_configs', return_value=[real_config_path]):
            # Load the actual config file
            with open(real_config_path, 'r') as file:
                real_config = yaml.safe_load(file)
            
            with patch('src.main.load_crew_config', return_value=real_config):
                mock_agent_manager.create_agents.return_value = {agent: MagicMock() for agent in real_config['agents']}
                mock_task_manager.create_tasks.return_value = [MagicMock() for _ in real_config['tasks']]
                mock_crew_runner.run_crew.return_value = "Real config execution result"
                
                await async_main()
                
                # Assertions based on your real config
                assert mock_agent_manager.create_agents.call_count == 1
                assert mock_task_manager.create_tasks.call_count == 1
                assert mock_crew_runner.run_crew.call_count == 1

# Run the test
if __name__ == "__main__":
    pytest.main(["-v", "test_main.py"])