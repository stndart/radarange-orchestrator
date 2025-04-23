import os
from .tool_annotation import (
    FunctionDescription,
    Parameters,
    ParameterProperty,
    Tool,
    ToolResult,
    ToolType,
)

ls_tool_def = ToolType(
    type="function",
    function=FunctionDescription(
        name="list_directory",
        description="Lists all files and directories in a specified path",
        parameters=Parameters(
            type="object",
            properties={
                "path": ParameterProperty(
                    type="string",
                    description="Directory path to list. Defaults to current directory."
                ),
            },
            required=[],
        ),
    ),
)


def ls_tool_handler(path: str = '.') -> ToolResult:
    try:
        contents = os.listdir(path)
        return ToolResult(
            status="success",
            stdout="\n".join(contents),
            stderr="",
            returncode=0
        )
    except Exception as e:
        return ToolResult(
            status="error",
            stdout="",
            stderr=f"Listdir error: {e}\nTraceback: {e.__traceback__}",
            returncode=1
        )


ls_tool = Tool(definition=ls_tool_def, handler=ls_tool_handler)