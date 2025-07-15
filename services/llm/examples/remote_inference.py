from services.llm import llm, config

"""
Run models on remote lmstudio server
"""

config.LMSTUDIO_ADDRESS = "192.168.1.10"
m = llm(model='qwq-32b@q4_k_m', backend='lmstudio')

print(m.respond("Who are you?").content)