from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os 

# load api key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.environ.get('GOOGLE_CSE_ID')

search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
tools = [
    Tool(
        name="검색",
        func=search.run,
        description="질문에 대답해야 할 필요가 있을 때 매우 유용함",
    )
]

prefix = """사용자와 대화한 후, 최선을 다하여 주어진 질문에 답하세요. 당신은 다음과 같은 도구를 사용할 수 있습니다:"""
suffix = """시작!"

{chat_history}
질문: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo-16k-0613'), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

print(agent_chain.run(input=""))