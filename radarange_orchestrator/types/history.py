from typing import Callable, Literal, Optional, TypeVar

from pydantic import BaseModel

from ..types.tools import ToolRequest

MessageRole = Literal['system', 'user', 'assistant', 'tool']
FinishReason = Literal['stop', 'length', 'tool_call']


class SystemPrompt(BaseModel):
    role: Literal['system'] = 'system'
    content: str


class UserMessage(BaseModel):
    role: Literal['user'] = 'user'
    content: str


class AssistantMessage(BaseModel):
    role: Literal['assistant'] = 'assistant'
    content: str
    finish_reason: FinishReason
    tool_calls: list[ToolRequest] = []


class ToolCallResponse(BaseModel):
    role: Literal['tool'] = 'tool'
    content: str
    tool_call_id: str


AnyChatMessage = SystemPrompt | UserMessage | AssistantMessage | ToolCallResponse


class AssistantMessageFragment(BaseModel):
    content: str


_T = TypeVar('_T', bound=AnyChatMessage)
MessageHandler = Callable[[_T], Optional[_T]]


def EmptyMessageHandler(response: AssistantMessage):
    pass
