from typing import TypedDict, Literal, Callable, Optional

class ParameterProperty(TypedDict):
    type: Literal["string", "number", "boolean", "integer", "array", "object"]
    description: str

class Parameters(TypedDict):
    type: Literal["object"]
    properties: dict[str, ParameterProperty]
    required: list[str]

class FunctionDescription(TypedDict):
    name: str
    description: str
    parameters: Parameters

class ToolType(TypedDict):
    type: Literal["function"]
    function: FunctionDescription

class ToolResult(TypedDict):
    status: str
    stdout: str
    stderr: str
    returncode: int
    
ToolHandler = Callable[..., ToolResult]

class Tool(TypedDict):
    definition: ToolType
    handler: ToolHandler

def get_tool_defs(tools: list[Tool]) -> list[ToolType]:
    return [t["definition"] for t in tools]

def get_tool_handler(tools: list[Tool], name: str) -> Optional[ToolHandler]:
    return next((tool["handler"] for tool in tools if tool["definition"]["function"]["name"] == name), None)

class LLMMessage(TypedDict):
    role: str
    content: str
