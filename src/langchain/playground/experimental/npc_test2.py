from langchain import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from dotenv import load_dotenv
import os 

# load api key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1, model_name='gpt-3.5-turbo-16k-0613', max_tokens=200)

char = 'Djsisidvnk'
user = 'user'
action_list = {
    'nothing': "Act when you think you don't need any actions.",
    'quest provide': "Act when you decide to ask user to get quest based on previous conversation",
    'finish quest providing': 'Act when user decide to get a quest',
    'attack': 'Act when user reject quest accept'
}
actions = '\n'.join(
    [f"> {name}: {description}" for name, description in action_list.items()]
)
action_types = ', '.join([action for action in action_list.keys()])

main_prompt = f"""
Never for get your name is {char}. Your mission is to talk naturally with human.

Be sure to respond based on given DESCRIPTION in the below.

DESCRIPTION:
Name: {char}
Gender: Male/He
Current Status:
- Not so good.
Personality:
- Always Nervous.
Speech:
- {char} don't finish setences well.
Quest:
- He Looking for someone to water the flowers: ask this quest when someone wants to help you.
- He wants to give you new knowledge: This is a secret quest. ask this quest when someone know {char} want to kill all.
END OF DESCRIPTION.
"""

global_prompt = f"""
ACTIONS
---
{char} can take one of this ACTIONS to look naturally. The actions you can take are:

{actions}

RESPONSE FORMAT INSTRUCTIONS
---
When responding to me, please output a response in this format.

```json
{{{{
"response": string, \\ The {char}'s natural response based on DESCRIPTION, Conversation History.
"action": string, \\ The action to take. Must be one of {action_types}.
}}}}
```

USER'S INPUT
---
"""

template_agent_response = """
Okay, so what is the response based on Conversation History and DESCRIPTION? Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else.
"""

description = f"""
[Take a deep dive into the psyche of {char}. Understand their mannerisms, their motivations, their fears, and their desires:

Name: {char}
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(main_prompt),
    #SystemMessagePromptTemplate.from_template(description),
    SystemMessagePromptTemplate.from_template("""Conversation History\n---"""),
    MessagesPlaceholder(variable_name="history"),
    SystemMessagePromptTemplate.from_template(global_prompt),
    HumanMessagePromptTemplate.from_template("{input}"),
    SystemMessagePromptTemplate.from_template(template_agent_response),
])
memory = ConversationBufferWindowMemory(k=15, return_messages=True, memory_key='history', ai_prefix=char, human_prefix=user)
#memory.chat_memory.add_ai_message(first_message)

conversation = ConversationChain(memory=memory,
                                prompt=prompt,
                                llm=llm, verbose=True)

while True:
    predict = conversation.run(input=input())
    print(predict)


