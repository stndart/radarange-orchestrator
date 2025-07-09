import json
import re

from radarange_orchestrator.tools import ToolCall, InvalidToolCall


def remove_think_block(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


tool_call_counter: int = 0


def extract_tool_calls(
    text: str, skip_reasoning: bool = True
) -> list[ToolCall | InvalidToolCall]:
    """
    Extracts last <tool_call></tool_call> block from text and optionally executes it
    """
    global tool_call_counter

    def invalid_tool_call(
        tool_call_counter: int, name: str = 'invalid'
    ) -> InvalidToolCall:
        return InvalidToolCall(name=name, args='', id=f'invalid_{tool_call_counter}')

    # If we need to skip <think> blocks, remove them from the text
    if skip_reasoning:
        # This will remove everything from <think> to </think>, including newlines.
        text = remove_think_block(text)

    tag_begin, tag_end = '<tool_call>', '</tool_call>'
    results: list[ToolCall | InvalidToolCall] = []
    for tool_block in re.findall(
        rf'{re.escape(tag_begin)}(.*?){re.escape(tag_end)}', text, re.DOTALL
    ):
        tool_call_counter += 1

        try:
            # Parse the JSON content of the tool call
            tool_data = json.loads(tool_block)
        except json.JSONDecodeError:
            results.append(invalid_tool_call(tool_call_counter))
        else:
            tool_name = tool_data.get('name')
            args = tool_data.get('arguments', {})
            if not tool_name:
                results.append(invalid_tool_call(tool_call_counter))
                continue

            results.append(
                ToolCall(
                    name=tool_name,
                    args=args,
                    id=f'{tool_name}_{tool_call_counter}',
                    type='tool_call',
                )
            )

    return results
