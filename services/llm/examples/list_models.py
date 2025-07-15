from services.llm.llm_backend import Model

"""
Lists available models for chosen backend.
Llama_cpp backend lists all .gguf files located in models directory
"""

print('Local available models:')
for model in Model.available_models(backend='llama_cpp'):
    print(model)
print()

print('Available models on remote:')
for model in Model.available_models(backend='lmstudio'):
    print(model)