import radarange_orchestrator as rorch

model = rorch.llm(backend='remote')
chat = rorch.Chat(tools=[rorch.tools.net_tool, rorch.tools.download_tool, rorch.tools.pdf_tool])
chat.add_user_message("""
Перескажи мне недавно вышедшую статью S. Drofa про бихроматическое возбуждение переходов в NV-центрах в алмазе и КПН.
Статья вышла недавно, на английском языке, и есть на arxiv.org. Скачай pdf файл статьи и прочитай её целиком.
Также найди самые важные сведения, которых нет в аннотации, но есть в полном тексте статьи.
""")
message = model.act(chat, on_message=chat.append, max_prediction_rounds=10, max_tokens_per_message=10000)

# shows correctly in jupyter notebooks
md = chat.show_final_answer()

# for CLI
print(md.data)