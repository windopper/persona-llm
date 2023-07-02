from generative_agent.GenerativeAgent import GenerativeAgent
from datetime import datetime
import guidance
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# guidance.llm = guidance.llms.OpenAI("text-davinci-003", api_key=OPENAI_API_KEY)
# prompt = guidance('''The best thing about the beach is {{~gen 'best' temperature=0.7 max_tokens=7}}''')
# res = prompt()
# print(res)

# exit()

guidance.llm = guidance.llms.OpenAI(model="text-davinci-002", api_key=OPENAI_API_KEY)
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

now = datetime.now()
new_time = now.replace(hour=7, minute=25)
description = "Sam is a Ph.D student, his major is CS;Sam likes computer;Sam lives with his friend, Bob;Sam's farther is a doctor;Sam has a dog, named Max"
sam = GenerativeAgent(
    guidance=guidance,
    name="Sam",
    age=23,
    description=description,
    traits="funny, like football, play CSGO",
    embeddings_model=embeddings_model,
    current_time=new_time,
)

sam.update_status()

print(sam.plan)
