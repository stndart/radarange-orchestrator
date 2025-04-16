import re
from IPython.display import display, Markdown, HTML

def display_thoughts(text: str):
  prefix = '''
  <style>
    think {
      display: block;
      background: #333;
      border-left: 4px solid #007acc;  # VS Code's blue accent color
      margin: 10px 0;
      padding: 10px;
      color: #fff;
      white-space: pre-wrap;  # Preserve line breaks
    }
    tool_call {
      display: block;
      background: #131;
      border-left: 4px solid #007acc;  # VS Code's blue accent color
      margin: 10px 0;
      padding: 10px;
      color: #fbf;
      white-space: pre-wrap;  # Preserve line breaks
    }
  </style>
  '''
  display(HTML(prefix + text))

def show_final_answer(messages: list[dict], hide_reasoning: bool = True):
  last_message = messages[-1]['choices'][0]['message']['content']
  text = re.sub(r'<think>.*?</think>', '', last_message, flags=re.DOTALL)
  display(Markdown(text))