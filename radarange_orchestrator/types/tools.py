from typing import Callable, Literal

from pydantic import BaseModel

ParameterType = Literal['string', 'number', 'boolean', 'integer', 'array', 'object']


class ParameterProperty(BaseModel):
    type: ParameterType
    description: str


class Parameters(BaseModel):
    type: Literal['object']
    properties: dict[str, ParameterProperty]
    required: list[str]


EmptyParameters = Parameters(type='object', properties=dict(), required=[])


class FunctionDescription(BaseModel):
    name: str
    description: str
    parameters: Parameters = EmptyParameters


class ToolDef(BaseModel):
    type: Literal['function']
    function: FunctionDescription


ToolResultStatus = Literal['success', 'error']


class ToolResult(BaseModel):
    status: ToolResultStatus
    stdout: str = ''
    stderr: str = ''
    returncode: int = 0


ToolHandler = Callable[..., ToolResult]


class Tool(BaseModel):
    definition: ToolDef
    handler: ToolHandler


def DefaultToolHandler(*args, **kwarge) -> ToolResult:
    return ToolResult(status='error', stdout='', stderr='No tool found', returncode=-1)


def get_tool_handler(tools: list[Tool], name: str) -> ToolHandler:
    return next(
        (tool.handler for tool in tools if tool.definition.function.name == name),
        DefaultToolHandler,
    )


class ToolRequest(BaseModel):
    id: str
    name: str
    arguments: str
