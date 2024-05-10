#!/usr/bin/env python
# coding: utf-8

# # Lesson 3: Building an Agent Reasoning Loop

# ## Setup

# In[1]:


from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()


# In[2]:


import nest_asyncio
nest_asyncio.apply()


# ## Load the data

# To download this paper, below is the needed code:
# 
# #!wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
# 
# **Note**: The pdf file is included with this lesson. To access it, go to the `File` menu and select`Open...`.

# ## Setup the Query Tools

# In[3]:


from utils import get_doc_tools

vector_tool, summary_tool = get_doc_tools("metagpt.pdf", "metagpt")


# ## Setup Function Calling Agent

# In[5]:


from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)


# In[6]:


from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)


# In[7]:


response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)


# In[8]:


print(response.source_nodes[0].get_content(metadata_mode="all"))


# In[9]:


response = agent.chat(
    "Tell me about the evaluation datasets used."
)


# In[10]:


response = agent.chat("Tell me the results over one of the above datasets.")


# ## Lower-Level: Debuggability and Control

# In[11]:


agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)


# In[26]:


task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other and  also explain me every thing about how agents communicate with each other like shared message tools and subscription mechanisms"
)


# In[27]:


step_output = agent.run_step(task.task_id)


# In[19]:


completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
print(completed_steps[0].output.sources[0].raw_output)


# In[28]:


upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]


# In[29]:


step_output = agent.run_step(
    task.task_id, input="What about how agents share information?"
)


# In[35]:


step_output = agent.run_step(task.task_id)
print(step_output.is_last)


# In[ ]:


response = agent.finalize_response(task.task_id)


# In[34]:


print(str(response))

