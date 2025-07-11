from radarange_orchestrator import llm

m = llm(model='QwQ-32B-Q4_K_M.gguf', backend='llama_cpp')

print(m.respond("Who are you?").content)