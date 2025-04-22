import re, json
from ..tools.tool_annotation import ToolCall, ToolCallFunction

tool_call_counter: int = 0
def extract_tool_calls(text: str, skip_reasoning: bool = True) -> list[ToolCall]:
    """
    Extracts last <tool_call></tool_call> block from text and optionally executes it
    """
    global tool_call_counter
    def invalid_tool_call(tool_call_counter: int, name: str = 'invalid') -> ToolCall:
        return ToolCall(
            id = f'invalid_{tool_call_counter}',
            type = 'function',
            function = ToolCallFunction(
                name = name,
                arguments = ''
            )
        )
    
    # If we need to skip <think> blocks, remove them from the text
    if skip_reasoning:
        # This will remove everything from <think> to </think>, including newlines.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    tag_begin, tag_end = "<tool_call>", "</tool_call>"
    results: list[ToolCall] = []
    for tool_block in re.findall(rf"{re.escape(tag_begin)}(.*?){re.escape(tag_end)}", text, re.DOTALL):
        tool_call_counter += 1

        try:
            # Parse the JSON content of the tool call
            tool_data = json.loads(tool_block)
        except json.JSONDecodeError:
            results.append(invalid_tool_call(tool_call_counter))
        else:
            tool_name = tool_data.get('name')
            arguments = tool_data.get('arguments', {})
            if not tool_name:
                results.append(invalid_tool_call(tool_call_counter))
                continue
            
            results.append(ToolCall(
                id = f'{tool_name}_{tool_call_counter}',
                type = 'function',
                function = ToolCallFunction(
                    name = tool_name,
                    arguments = json.dumps(arguments)
                )
            ))
    
    return results