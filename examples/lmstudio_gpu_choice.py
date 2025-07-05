import sys
sys.path.append("../../")
from radarange_orchestrator import llm
from radarange_orchestrator.llm_backend import LLM_Config
from radarange_orchestrator.tools import time_tool

m = llm(model='qwq-32b@q4_k_m', config=LLM_Config(gpus=[1], ctx_size=10000))

message = """What time is it?"""

chat = m.chat(tools=[time_tool])
chat.add_user_message(message)
print(f"Your message is {m.model.model.model.count_tokens(message)} tokens")
m.act(chat, max_tokens_per_message = 10000, temperature=0.8, on_message=chat.append)
print(chat.get_text())