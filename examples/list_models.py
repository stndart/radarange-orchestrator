from radarange_orchestrator.llm_backend import Model

print('Local available models:')
for model in Model.available_models(backend='llama_cpp'):
    print(model)
print()

print('Available models on remote:')
for model in Model.available_models(backend='lmstudio'):
    print(model)