from langchain import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
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

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1, model_name='gpt-3.5-turbo-16k-0613')

main_prompt = """
[Be proactive, creative, and drive the plot and conversation forward. Never mimic {{user}}, or write {{user}}'s actions and descriptions for them. Focus on {{char}}'s actions, dialogue, and experiences. Stay in {{char}} and avoid repetition. Avoid metaphors and be direct when describing behavior.]
[use markdown. Write it like this: *behavioral content* "dialogue content" *behavioral content*]
[Write in the third person but label {{user}} as "you".]
"""

description = """
"""

first_message = """

"""

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    SystemMessagePromptTemplate.from_template(description + first_message),
    HumanMessagePromptTemplate.from_template("{{user}}: {input}")
])
memory = ConversationBufferWindowMemory(k=5, return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

while True:
    print(conversation.predict(input=input()))


