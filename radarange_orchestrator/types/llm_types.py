from typing import Literal, Optional, Callable, TypeVar
from pydantic import BaseModel

from ..tools.tool_annotation import ToolCall
from llama_cpp import LLAMA_SPLIT_MODE_NONE, LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_ROW

class LlmConfig(BaseModel):
    gpus: list[int] = [0, 1]
    ctx_size: int = 0
    split_mode: Literal[LLAMA_SPLIT_MODE_NONE, LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_ROW] # type: ignore

MessageRole = Literal["system", "user", "assistant", "tool", "function"]
class ChatMessage(BaseModel):
    role: Literal["system", "user"]  # all other roles require more fields
    content: str

class ToolCallResponse(BaseModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str

class Response(BaseModel):
    role: Literal["assistant"]
    content: str
    tool_calls: list[ToolCall] = []
    finish_reason: Optional[str]

class ResponseStream:
    pass

MessageType = ChatMessage | ToolCallResponse | Response

_T = TypeVar("_T", bound=MessageType)
MessageHandler = Callable[[_T], Optional[_T]]
def EmptyMessageHandler(response: Response):
    pass