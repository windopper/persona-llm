from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os 

# load api key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)

from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType

def open_the_door(password: str) -> float:
    """open the door when knowing password"""
    if password == "34x34":
        return "success"
    
    return "failed"

def get_password() -> str:
    """get password to open the door"""
    return "34x34"

tool1 = StructuredTool.from_function(open_the_door)
tool2 = StructuredTool.from_function(get_password)

agent_executor = initialize_agent(
    [tool1, tool2],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent_executor.run("i want to open the door")