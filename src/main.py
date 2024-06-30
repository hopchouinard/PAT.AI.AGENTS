import yaml
import os
import sys
import logging
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama
from search_tool import create_search_tool
from sec_tools import SECTools

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.debug(f"Current directory: {current_dir}")

# Get the parent directory (project root)
project_root = os.path.dirname(current_dir)
logger.debug(f"Project root: {project_root}")

# Construct the path to the .env file
env_path = os.path.join(project_root, '.env')
logger.debug(f".env file path: {env_path}")

# Check if .env file exists
if os.path.exists(env_path):
    logger.debug(f".env file found at {env_path}")
else:
    logger.error(f".env file not found at {env_path}")

# Load environment variables from the project root
load_dotenv(env_path)

# Access API keys
SEC_API_KEY = os.getenv('SEC_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

logger.debug(f"SEC_API_KEY found: {'Yes' if SEC_API_KEY else 'No'}")
logger.debug(f"SERPER_API_KEY found: {'Yes' if SERPER_API_KEY else 'No'}")

# Validate that necessary API keys are present
if not SEC_API_KEY or not SERPER_API_KEY:
    logger.error("Missing required API keys. Please check your .env file.")
    sys.exit(1)

# Set API keys for tools that need them
os.environ['SERPER_API_KEY'] = SERPER_API_KEY

def load_crew_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_agents(config, ollama_llm):
    agents = {}
    search_tool = create_search_tool()
    sec_tools = [SECTools.search_10q, SECTools.search_10k]

    for agent_name, agent_config in config['agents'].items():
        tools = []
        if agent_config.get('use_search_tool', False):
            tools.append(search_tool)
        if agent_config.get('use_sec_tools', False):
            tools.extend(sec_tools)
        
        agents[agent_name] = Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            verbose=agent_config.get('verbose', True),
            allow_delegation=agent_config.get('allow_delegation', False),
            tools=tools,
            llm=ollama_llm
        )
    return agents

def create_tasks(config, agents):
    tasks = []
    
    # Get input for the crew
    if 'company_name' in config['tasks'][0]['description']:
        company_name = input("Enter the company name for analysis: ")
        variable = {'company_name': company_name}
    elif 'ai_prompt' in config['tasks'][0]['description']:
        ai_prompt = input("Enter the AI prompt for analysis: ")
        variable = {'ai_prompt': ai_prompt}
    else:
        variable = {}
    
    # Create tasks
    for task_config in config['tasks']:
        task_description = task_config['description'].format(**variable)
        tasks.append(Task(
            description=task_description,
            agent=agents[task_config['agent']],
            expected_output=task_config['expected_output']
        ))
    
    return tasks

def main():
    # Path to the 'crew' folder
    crew_folder = os.path.join(project_root, 'crew')
    
    # List available YAML files in the 'crew' folder
    yaml_files = [f for f in os.listdir(crew_folder) if f.endswith('.yaml')]
    
    if not yaml_files:
        logger.error(f"No YAML files found in the '{crew_folder}' folder.")
        sys.exit(1)
    
    print("Available crew configurations:")
    for i, file in enumerate(yaml_files, 1):
        print(f"{i}. {file}")
    
    # Let the user choose a configuration
    choice = int(input("Enter the number of the configuration you want to run: ")) - 1
    chosen_file = yaml_files[choice]
    
    # Load the chosen configuration
    config = load_crew_config(os.path.join(crew_folder, chosen_file))
    
    # Initialize Ollama with the specified model
    try:
        llm_model = config.get('llm_model', 'llama3:latest')  # Default to 'llama3:latest' if not specified
        logger.info(f"Initializing Ollama with model: {llm_model}")
        ollama_llm = Ollama(model=llm_model)
    except KeyError as e:
        logger.error(f"Error in configuration file: Missing key {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing Ollama: {e}")
        sys.exit(1)
    
    # Create agents and tasks
    try:
        agents = create_agents(config, ollama_llm)
        tasks = create_tasks(config, agents)
    except KeyError as e:
        logger.error(f"Error in configuration file: Missing key {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating agents or tasks: {e}")
        sys.exit(1)
    
    # Create and run the crew
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        verbose=2,
        process=config.get('process', 'sequential')
    )
    
    try:
        result = crew.kickoff()
        logger.info("Crew's work completed successfully")
        print("Crew's work result:")
        print(result)
    except Exception as e:
        logger.error(f"Error during crew kickoff: {e}")

if __name__ == "__main__":
    main()