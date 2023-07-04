PROMPT_ADDMEM = """
{{#system~}}
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory. Respond with a single integer.
Memory: {{memory_content}}

Rating: 
{{~/system}}
{{#assistant~}}
{{gen 'rate' stop='\\n'}}
{{~/assistant}}
"""

PROMPT_ADDMEMS = """
{{#system~}}
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.\\
Always answer with only a list of numbers.
If just given one memory still respond in a list.
Memories are separated by semi colans (;)
Memories: {{memory_content}}

Rating: 
{{~/system}}
{{#assistant~}}
{{gen 'rate' stop='\\n'}}
{{~/assistant}}
"""

PROMPT_SUMMARIZE = """
{{#system~}}
How would you summarize {{name}}'s core characteristics given the following statements:
{{relevant_memories}}
Do not embellish

Summary:
{{~/system}}
{{#assistant~}}
{{gen 'summary' temperature=0.5}}
{{~/assistant}}
"""

PROMPT_SALIENT = """
{{#user~}}
{{recent_memories}}

Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?
{{~/user}}
{{#assistant~}}
{{gen 'items'}}
{{~/assistant}}"""

PROMPT_INSIGHTS = """
{{#user~}}
{{statements}}

What 3 high-level insights can you infer from the above statements?
{{~/user}}
{{#assistant~}}
{{gen 'items' temperature=0.5}}
{{~/assistant}}"""

PROMPT_PLAN = """
{{#system~}}
Name: {{name}}. 
Innate traits: {{traits}}
The following is your description: {{summary}}
What is your goal for today? Write it down in an hourly basis, starting at {{now}}. 
Generate 5~8 plans by writing only one or two very short sentences.
Be very brief. Use at most 50 words every plan.
output format:
HH:MM - HH:MM) what to do
{{~/system}}
{{#assistant~}}
{{gen 'plans' temperature=0.5 max_tokens=500}}
{{~/assistant}}
"""

PROMPT_RECURSIVELY_DECOMPOSED = """
{{#system~}}
You are {{name}}.
The following is your description: {{summary}}
current plans: {{plans}}
Decomposing current plans into 5~15 minute chunks.
Be very brief. Use at most 50 words every plan.
{{~/system}}
{{#assistant~}}
{{gen 'plans' temperature=0.5 max_tokens=500}}
{{~/assistant}}
"""

PROMPT_CONTEXT = """
{{#system~}}
Summarize those statements.

Example:
Given statements:
- Gosun has power, but he is struggling to deal with living costs
- Gosun see Max is sick
- Gosun has a dog, named Max
- Bob is in dangerous

Focus on Gosun and Max and statement: "Max is sick".

Summary: Gosun has a dog named Max, who is sick. Gosun has power, but he is struggling to deal with living costs. His friend, Bob, is in dangerous.

Given statements:
{{statements}}

Summarize those statements, focus on {{name}} and {{observed_entity}} and statement: "{{entity_status}}".

Summary:
{{~/system}}
{{#assistant~}}
{{gen 'context' max_tokens=300 stop='\\n'}}
{{~/assistant}}"""

PROMPT_REACT = """
{{#system~}}
{{summary}}

It is {{current_time}}.
{{name}}'s status: {{status}}
Observation: {{observation}}

Summary of relevant context from {{name}}'s memory: {{context}}

Should {{name}} react to the observation, and if so, what would be an appropriate reaction?

Reaction: select Yes or No
{{~/system}}

{{#assistant~}}
{{gen 'reaction'}}
{{~/assistant}}

{{#system~}}
Appropriate reaction: 
{{~/system}}

{{#assistant~}}
{{gen 'result' temperature=0.5 stop='\\n'}}
{{~/assistant}}
"""

PROMPT_REPLAN = """
{{#system~}}

{{summary}}.
It is {{current_time}} now. Please make a plan from now for {{name}} in broad stroke given his/her reaction.

It is {{current_time}} now. 
{{name}}'s status: {{status}}
Observation: {{observation}}
{{name}}'s reaction: {{reaction}}

Generate 5 plans by writing only one or two very short sentences.
Be very brief. Use at most 50 words every plan.
output format:
HH:MM - HH:MM) what to do
{{~/system}}
{{#assistant~}}
{{gen 'plans' temperature=0.5}}
{{~/assistant}}
"""

PROMPT_DIALOGUE = """
{{#system~}}
{{summary}}

It is {{current_time}}.
{{name}}'s status:{{status}}
Observation: {{observation}}

Summary of relevant context from {{name}}'s memory: {{context}}

Example of dialogue:
A: Wow, it is a nice haircut
B: Thank you! How is your school project?
A: I'm still trying.
B: Good luck.

{{~/system}}
{{#user~}}
{{name}}'s reaction: {{reaction}}
What would {{name}} say to {{observed_entity}}? Make a short dialogue.

Here is the short dialogue:
{{~/user}}
{{#assistant~}}
{{gen 'dialogue' temperature=0.5}}
{{~/assistant}}
"""

PROMPT_INTERVIEW = """
{{#system~}}
{{summary}}

It is {{current_time}}.
{{name}}'s status:{{status}}

Summary of relevant context from {{name}}'s memory:
{{context}}
{{~/system}}
{{#user~}}
The {{user}} say "{{question}}". What should {{name}} response?

Here is the response from {{name}}: 
{{~/user}}
{{#assistant~}}
{{gen 'response' temperature=0.5}}
{{~/assistant}}"""
