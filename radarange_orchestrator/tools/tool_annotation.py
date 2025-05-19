from typing import TypedDict, Literal, Callable, Optional
from pydantic import BaseModel
from llama_cpp.llama_types import ChatCompletionTool
import json

ParameterType = Literal["string", "number", "boolean", "integer", "array", "object"]
class ParameterProperty(BaseModel):
    type: ParameterType
    description: str

class Parameters(BaseModel):
    type: Literal["object"]
    properties: dict[str, ParameterProperty]
    required: list[str]

EmptyParameters = Parameters(
    type = "object",
    properties = dict(),
    required = []
)
class FunctionDescription(BaseModel):
    name: str
    description: str
    parameters: Parameters = EmptyParameters

class ToolType(BaseModel):
    type: Literal["function"]
    function: FunctionDescription

class ToolResult(BaseModel):
    status: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0

def tool_result_to_str(result: ToolResult) -> str:
    return result.model_dump_json()
    
ToolHandler = Callable[..., ToolResult]

class Tool(BaseModel):
    definition: ToolType
    handler: ToolHandler

def DefaultToolHandler(*args, **kwarge) -> ToolResult:
    return ToolResult(status="error", stdout="", stderr="No tool found", returncode=-1)

def get_tool_defs(tools: list[Tool]) -> list[ChatCompletionTool]:
    return [json.loads(t.definition.model_dump_json()) for t in tools]

def get_tool_handler(tools: list[Tool], name: str) -> ToolHandler:
    return next((tool.handler for tool in tools if tool.definition.function.name == name), DefaultToolHandler)

class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolCallFunction