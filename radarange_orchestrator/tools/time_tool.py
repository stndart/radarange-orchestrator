from ..types.tools import FunctionDescription, Parameters, Tool, ToolResult, ToolDef

import datetime

tool_def = ToolDef(
    type = "function",
    function = FunctionDescription(
        name = "time_tool",
        description = "Get's actual current time in local timezone",
        parameters = Parameters(
            type = "object",
            properties = dict(),
            required = []
        )
    )
)

def handle_tool() -> ToolResult:
    result = ToolResult(
        status = "success",
        stdout = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        stderr = "",
        returncode = 0
    )
    return result

time_tool = Tool(
    definition = tool_def,
    handler = handle_tool
)