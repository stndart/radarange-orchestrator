from ..types.tools import FunctionDescription, Parameters, ParameterProperty, Tool, ToolResult, ToolDef

tool_def = ToolDef(
    type = "function",
    function = FunctionDescription(
        name = "search_internet",
        description = "",
        parameters = Parameters(
            type = "object",
            properties = {
                "param1": ParameterProperty(
                    type = "string",
                    description = "param 1"
                ),
            },
            required = []
        )
    )
)

def handle_tool() -> ToolResult:
    result = ToolResult(
        status = "success",
        stdout = "",
        stderr = "",
        returncode = 0
    )
    return result

tool = Tool(
    definition = tool_def,
    handler = handle_tool
)