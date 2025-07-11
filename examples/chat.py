from radarange_orchestrator import llm, config, Chat

config.LMSTUDIO_ADDRESS = "192.168.1.10"
m = llm(model='qwq-32b@q4_k_m', backend='lmstudio')

chat = Chat()
chat.add_user_message("What's the difference betweeen LLM and VLM?")

r = m.respond(chat)
print(r.content)

chat.add_user_message("Is Qwen2.5-VL a VLM?")

r = m.respond(chat)
print(chat[-1].content)