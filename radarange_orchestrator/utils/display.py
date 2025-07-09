import json

from IPython.display import HTML, Markdown, display
from ..chat.messages import AnyCompleteMessage, ToolMessage
from ..tools import ToolMessage
from .extract_tool_calls import remove_think_block


def display_thoughts(text: str):
    prefix = """
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
    """
    display(HTML(prefix + text))


def show_final_answer(messages: list[dict], hide_reasoning: bool = True):
    last_message = messages[-1]['choices'][0]['message']['content']
    text = remove_think_block(last_message)
    display(Markdown(text))


def display_message(
    message: AnyCompleteMessage,
    skip_reasoning: bool = False,
    truncate_tool_response: bool = True,
) -> HTML:
    background = '#1e1e1e'
    message.type
    if message.type == 'tool':
        background = '#03074a'
    elif message.type == 'ai':
        background = '#360228'

    prefix = """
    <style>
        body, .container {
            background: %s;  /* Dark background like VS Code */
            color: #ccc;     /* Light grey text */
        }
        think {
            display: block;
            background: #333;
            border-left: 4px solid #007acc;
            margin: 10px 0;
            padding: 10px;
            color: #fff;
            white-space: pre-wrap;
        }
        tool_call {
            display: block;
            background: #131;
            border-left: 4px solid #007acc;
            margin: 10px 0;
            padding: 10px;
            color: #fbf;
            white-space: pre-wrap;
        }
    </style>
    """ % (background)

    text = message.content
    if message.type == 'ai' and skip_reasoning:
        text = remove_think_block(text)

    return HTML(prefix + f'<body><div class="container">{text}</div></body>')
