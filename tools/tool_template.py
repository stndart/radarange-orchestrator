from .tool_annotation import Tool, ToolResult, ToolType

tool_def = ToolType(**{
    "type": "function",
    "function": {
        "name": "search_internet",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": []
        }
    }
})

def handle_tool() -> ToolResult:
    result = ToolResult(**{"status": "success", "stdout": "", "stderr": "", "returncode": 0})
    return result

tool = Tool(**{
    "definition": tool_def,
    "handler": handle_tool
})