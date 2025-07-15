from services.llm import llm, LLM_Config

"""
Choose cuda gpu to load model weights onto.
You may use any number of gpus, as well as none (in this case, cpu is used)
"""

conf = LLM_Config(gpus=[1], ctx_size=10000)
m = llm(model='qwq-32b@q4_k_m', config=conf)

print(m.respond("Who are you?").content)