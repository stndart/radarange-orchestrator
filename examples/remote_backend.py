import radarange_orchestrator as rorch

model = rorch.llm(backend='remote')
message = model.respond("What's the capital of Paris?")
print(f'{message.role}: {message.content}')