import lmstudio as lms
from radarange_orchestrator import Chat
from radarange_orchestrator.backend.lmstudio_remote_model import convert_chat

client = lms.Client('95.165.10.219:1234')
model = client.llm.model('qwq-32b@q4_k_m', ttl=300)

chat = Chat()
chat.add_user_message("What's the capital of Paris?")

for token in model.respond_stream(history=convert_chat(chat)):
    print(token.content, end='', flush=True)
print()
