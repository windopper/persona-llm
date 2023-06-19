from langchain import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from dotenv import load_dotenv
import os 

# load api key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)

description = """
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(description),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{{사용자}}: {input}")
])
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

while True:
    print(conversation.predict(input=input()))


