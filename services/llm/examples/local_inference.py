from services.llm import llm

"""
Run models locally via llama_cpp
"""

m = llm(model='QwQ-32B-Q4_K_M.gguf', backend='llama_cpp')

print(m.respond("Who are you?").content)