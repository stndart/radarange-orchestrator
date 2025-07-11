from radarange_orchestrator import llm, LLM_Config

conf = LLM_Config(gpus=[1], ctx_size=10000)
m = llm(model='qwq-32b@q4_k_m', config=conf)

print(m.respond("Who are you?").content)