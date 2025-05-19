import json
from IPython.display import display, Markdown, HTML
from requests import Response

from radarange_orchestrator.llm_types import MessageType, ToolCallResponse
from radarange_orchestrator.tools.tool_annotation import ToolResult
from radarange_orchestrator.utils.extract_tool_calls import remove_think_block


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
    last_message = messages[-1]["choices"][0]["message"]["content"]
    text = remove_think_block(last_message)
    display(Markdown(text))


def display_message(
    message: MessageType,
    skip_reasoning: bool = False,
    truncate_tool_response: bool = True,
) -> HTML:
    background = "#1e1e1e"
    if isinstance(message, ToolCallResponse):
        background = "#03074a"
    elif isinstance(message, Response):
        background = "#360228"

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
    if isinstance(message, Response) and skip_reasoning:
        text = remove_think_block(text)
    if isinstance(message, ToolCallResponse) and truncate_tool_response:
        obj = ToolResult(**json.loads(text))
        obj.stdout = obj.stdout[:200]
        text = obj.model_dump_json(indent=2)

    return HTML(prefix + f'<body><div class="container">{text}</div></body>')
