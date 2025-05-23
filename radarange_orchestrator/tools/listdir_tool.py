import os
from ..types.tools import (
    FunctionDescription,
    Parameters,
    ParameterProperty,
    Tool,
    ToolResult,
    ToolDef,
)

ls_tool_def = ToolDef(
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
    print(f"Called list_directory with path: {path}", flush=True)
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