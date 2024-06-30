# tests/unit/test_agent_manager.py

import pytest
from unittest.mock import Mock, patch
from src.agent_manager import AgentManager
from src.exceptions import AgentCreationError

@pytest.fixture
def agent_manager():
    mock_config = {'default_llm_model': 'test_model'}
    mock_ollama_llm = Mock()
    mock_search_tool = Mock()
    mock_sec_tools = Mock()
    return AgentManager(mock_config, mock_ollama_llm, mock_search_tool, mock_sec_tools)

@pytest.mark.asyncio
async def test_create_agents_success(agent_manager):
    mock_crew_config = {
        'agents': {
            'researcher': {
                'role': 'Researcher',
                'goal': 'Research',
                'backstory': 'Expert researcher',
                'use_search_tool': True,
                'use_sec_tools': False
            },
            'analyst': {
                'role': 'Analyst',
                'goal': 'Analyze',
                'backstory': 'Expert analyst',
                'use_search_tool': False,
                'use_sec_tools': True
            }
        }
    }

    with patch('src.agent_manager.Agent') as MockAgent:
        MockAgent.return_value = Mock()
        agents = await agent_manager.create_agents(mock_crew_config)

        assert len(agents) == 2
        assert 'researcher' in agents
        assert 'analyst' in agents
        assert MockAgent.call_count == 2

        # Check if the correct tools were assigned
        researcher_call = MockAgent.call_args_list[0]
        analyst_call = MockAgent.call_args_list[1]
        assert len(researcher_call.kwargs['tools']) == 1  # Only search tool
        assert len(analyst_call.kwargs['tools']) == 2  # Both SEC tools

@pytest.mark.asyncio
async def test_create_agents_missing_required_field(agent_manager):
    mock_crew_config = {
        'agents': {
            'researcher': {
                'role': 'Researcher',
                # 'goal' is missing
                'backstory': 'Expert researcher',
                'use_search_tool': True,
                'use_sec_tools': False
            }
        }
    }

    with pytest.raises(AgentCreationError) as exc_info:
        await agent_manager.create_agents(mock_crew_config)
    
    assert "Missing required configuration for agent researcher" in str(exc_info.value)

@pytest.mark.asyncio
async def test_create_agents_empty_config(agent_manager):
    mock_crew_config = {'agents': {}}

    agents = await agent_manager.create_agents(mock_crew_config)
    assert len(agents) == 0

@pytest.mark.asyncio
async def test_create_agents_invalid_tool_config(agent_manager):
    mock_crew_config = {
        'agents': {
            'researcher': {
                'role': 'Researcher',
                'goal': 'Research',
                'backstory': 'Expert researcher',
                'use_search_tool': 'invalid',  # Should be boolean
                'use_sec_tools': False
            }
        }
    }

    with patch('src.agent_manager.Agent') as MockAgent:
        MockAgent.return_value = Mock()
        agents = await agent_manager.create_agents(mock_crew_config)

        assert len(agents) == 1
        assert 'researcher' in agents
        # Check that no tools were assigned due to invalid config
        assert len(MockAgent.call_args.kwargs['tools']) == 0

@pytest.mark.asyncio
async def test_create_agents_all_tools(agent_manager):
    mock_crew_config = {
        'agents': {
            'super_agent': {
                'role': 'Super Agent',
                'goal': 'Do everything',
                'backstory': 'Expert in all fields',
                'use_search_tool': True,
                'use_sec_tools': True
            }
        }
    }

    with patch('src.agent_manager.Agent') as MockAgent:
        MockAgent.return_value = Mock()
        agents = await agent_manager.create_agents(mock_crew_config)

        assert len(agents) == 1
        assert 'super_agent' in agents
        # Check that all tools were assigned
        assert len(MockAgent.call_args.kwargs['tools']) == 3  # search_tool + 2 SEC tools

@pytest.mark.asyncio
async def test_create_agents_agent_creation_failure(agent_manager):
    mock_crew_config = {
        'agents': {
            'problematic_agent': {
                'role': 'Problematic',
                'goal': 'Cause issues',
                'backstory': 'Troublemaker',
                'use_search_tool': False,
                'use_sec_tools': False
            }
        }
    }

    with patch('src.agent_manager.Agent', side_effect=Exception("Agent creation failed")):
        with pytest.raises(AgentCreationError) as exc_info:
            await agent_manager.create_agents(mock_crew_config)
        
        assert "Failed to create agent problematic_agent" in str(exc_info.value)