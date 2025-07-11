from radarange_orchestrator import llm, config
from radarange_orchestrator.tools import time_tool

config.LMSTUDIO_ADDRESS = "192.168.1.10"
m = llm(model='qwq-32b@q4_k_m', backend='lmstudio')

print(m.act("What time is it?", tools=[time_tool]).content)