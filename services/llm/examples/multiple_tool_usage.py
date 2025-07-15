from services.llm import llm, config, Chat
from services.llm.tools import all_tools

"""
Both llama_cpp and lmstudio backends support multi-turn agent-like inferencing.
LLM can reason, call tools, parse their output, and answer with an answer.
Tools are langchain-compatible, see examples at radarange_orchestrator/tools/
"""

config.LMSTUDIO_ADDRESS = "192.168.1.10"
m = llm(model='qwq-32b@q4_k_m', backend='lmstudio')

prompt = """
Перескажи мне недавно вышедшую статью S. Drofa про бихроматическое возбуждение переходов в NV-центрах в алмазе и КПН. Статья вышла недавно, на английском языке, и есть на arxiv.org.
Скачай pdf файл статьи и прочитай её целиком. Также найди самые важные сведения, которых нет в аннотации, но есть в полном тексте статьи. Будь готов ответить на вопросы по статье.
"""
chat = Chat(tools=all_tools)
chat.add_user_message(prompt)

print(m.act(chat, max_prediction_rounds=10).content)