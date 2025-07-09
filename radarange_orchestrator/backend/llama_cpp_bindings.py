import json
from typing import overload

from llama_cpp import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionResponseMessage,
)
from llama_cpp.llama_types import (
    ChatCompletionMessageToolCall,
    ChatCompletionRequestMessage,
    ChatCompletionTool,
    CreateChatCompletionResponse,
)

from ..chat.chat import Chat
from ..chat.messages import (
    AIMessage,
    AnyCompleteMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from ..tools import Tool, ToolCall


@overload
def to_llama_message(message: AIMessage) -> ChatCompletionRequestAssistantMessage: ...
@overload
def to_llama_message(message: HumanMessage) -> ChatCompletionRequestUserMessage: ...
@overload
def to_llama_message(message: SystemMessage) -> ChatCompletionRequestSystemMessage: ...
@overload
def to_llama_message(message: ToolMessage) -> ChatCompletionRequestToolMessage: ...


def to_llama_message(message: AnyCompleteMessage) -> ChatCompletionRequestMessage:
    translate_roles = {
        'ai': 'assistant',
        'human': 'user',
        'system': 'system',
        'tool': 'tool',
    }

    result: ChatCompletionRequestMessage = {
        'role': translate_roles[message.type],
        'content': message.content,
    }
    if isinstance(message, ToolMessage):
        result['tool_call_id'] = message.tool_call_id
    elif isinstance(message, AIMessage):
        tool_calls: list[ChatCompletionMessageToolCall] = [
            {
                'id': tc['id'],
                'type': 'function',
                'function': {'name': tc['name'], 'arguments': tc['args']},
            }
            for tc in message.tool_calls
        ]
        result['tool_calls'] = tool_calls
    return result


@overload
def from_llama_message(message: ChatCompletionRequestAssistantMessage) -> AIMessage: ...
@overload
def from_llama_message(message: CreateChatCompletionResponse) -> AIMessage: ...
@overload
def from_llama_message(message: ChatCompletionResponseMessage) -> AIMessage: ...
@overload
def from_llama_message(message: ChatCompletionRequestUserMessage) -> HumanMessage: ...
@overload
def from_llama_message(
    message: ChatCompletionRequestSystemMessage,
) -> SystemMessage: ...
@overload
def from_llama_message(message: ChatCompletionRequestToolMessage) -> ToolMessage: ...


def from_llama_message(
    message: ChatCompletionRequestMessage
    | ChatCompletionResponseMessage
    | CreateChatCompletionResponse,
) -> AnyCompleteMessage:
    stop_reason = 'stop'

    # if isinstance(message, CreateChatCompletionResponse):
    if 'role' not in message:
        stop_reason = message['choices'][0]['finish_reason']
        message: ChatCompletionResponseMessage = message['choices'][0]['message']

    content = message['content']

    if message['role'] == 'assistant':
        # rebuild the tool_calls list
        tool_calls: list[ToolCall] = []
        for tc in message.get('tool_calls', []):
            tool_calls.append(
                {
                    'id': tc['id'],
                    'name': tc['function']['name'],
                    'args': tc['function']['arguments'],
                }
            )
        message = AIMessage(content=content, tool_calls=tool_calls)
        message.response_metadata['stop_reason'] = stop_reason
        return message
    elif message['role'] == 'user':
        return HumanMessage(content=content)
    elif message['role'] == 'system':
        return SystemMessage(content=content)
    elif message['role'] == 'tool':
        # ChatCompletionRequestToolMessage always has tool_call_id
        return ToolMessage(
            content=content,
            tool_call_id=message['tool_call_id'],
        )
    else:
        raise ValueError(f'Unknown role: {message["role"]}')


def to_llama_chat(chat: Chat) -> list[ChatCompletionRequestMessage]:
    return [to_llama_message(message) for message in chat]


def to_llama_tool(tool: Tool) -> ChatCompletionTool:
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description,
            'parameters': {'type': 'object', 'properties': tool.args},
        },
    }


def to_llama_tools(tools: list[Tool]) -> list[ChatCompletionTool]:
    return [to_llama_tool(tool) for tool in tools]


def to_llama_tool_call(tc: ToolCall) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        type='function',
        id=tc['id'],
        function={'name': tc['name'], 'arguments': json.dumps(tc['args'])},
    )
