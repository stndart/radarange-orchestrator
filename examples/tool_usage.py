from radarange_orchestrator import llm, config
from radarange_orchestrator.tools import time_tool

"""
Both llama_cpp and lmstudio backends support tool usage for models, that are trained for that.
Tools are langchain-compatible, see examples at radarange_orchestrator/tools/
"""

config.LMSTUDIO_ADDRESS = '192.168.1.10'
m = llm(model='qwq-32b@q4_k_m', backend='lmstudio')

print(m.act('What time is it?', tools=[time_tool]).content)
