from typing import TypedDict, Literal, Callable, Optional
from pydantic import BaseModel
from llama_cpp.llama_types import ChatCompletionTool
import json

class ParameterProperty(BaseModel):
    type: Literal["string", "number", "boolean", "integer", "array", "object"]
    description: str

class Parameters(BaseModel):
    type: Literal["object"]
    properties: dict[str, ParameterProperty]
    required: list[str]

class FunctionDescription(BaseModel):
    name: str
    description: str
    parameters: Parameters

class ToolType(BaseModel):
    type: Literal["function"]
    function: FunctionDescription

class ToolResult(BaseModel):
    status: str
    stdout: str
    stderr: str
    returncode: int
    
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

class LLMMessage(TypedDict):
    role: str
    content: str
