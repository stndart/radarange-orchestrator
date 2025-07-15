import json
from functools import wraps
from typing import Any, Literal, overload

import lmstudio as lms

from ..chat import (
    AIMessage,
    AnyCompleteMessage,
    Chat,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from ..tools import Tool


@overload
def to_lms_message(message: AIMessage) -> lms.AssistantResponse: ...
@overload
def to_lms_message(message: HumanMessage) -> lms.UserMessage: ...
@overload
def to_lms_message(message: SystemMessage) -> lms.SystemPrompt: ...
@overload
def to_lms_message(message: ToolMessage) -> lms.ToolResultMessage: ...


def to_lms_message(message: AnyCompleteMessage) -> lms.AnyChatMessage:
    if message.type == 'ai':
        return lms.AssistantResponse.from_dict(
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': message.content}],
            }
        )
    elif message.type == 'human':
        return lms.UserMessage.from_dict(
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': message.content}],
            }
        )
    elif message.type == 'system':
        return lms.SystemPrompt.from_dict(
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': message.content}],
            }
        )
    elif message.type == 'tool':
        return lms.ToolResultMessage.from_dict(
            {
                'role': 'tool',
                'content': [
                    {
                        'type': 'toolCallResult',
                        'content': message.content,
                        'toolCallId': message.tool_call_id,
                    }
                ],
            }
        )
    else:
        raise NotImplementedError(f'Unexpected message type: {message.type}')


@overload
def from_lms_message(message: lms.AssistantResponse) -> AIMessage: ...
@overload
def from_lms_message(message: lms.UserMessage) -> HumanMessage: ...
@overload
def from_lms_message(message: lms.SystemPrompt) -> SystemMessage: ...
@overload
def from_lms_message(message: lms.ToolResultMessage) -> list[ToolMessage]: ...


def from_lms_message(
    message: lms.AnyChatMessage,
) -> list[ToolMessage] | AIMessage | HumanMessage | SyntaxError:
    content = message.content
    if message.role == 'tool':
        tool_calls: list[ToolMessage] = []
        for token in content:
            tool_calls.append(
                ToolMessage(content=token.content, tool_call_id=token.tool_call_id)
            )
        return tool_calls

    text_data: list[str] = []
    for token in content:
        if token.type == 'text':
            text_data.append(token.text)
        elif token.type == 'toolCallRequest':
            text_data.append(json.dumps(token.tool_call_request.to_dict(), indent=2))

    if message.role == 'assistant':
        return AIMessage(content='\n'.join(text_data))
    elif message.role == 'user':
        return HumanMessage(content='\n'.join(text_data))
    elif message.role == 'system':
        return SystemMessage(content='\n'.join(text_data))
    else:
        raise NotImplementedError(f'Unexpected lms message role: {message.role}')


def from_lms_response(message: lms.PredictionResult) -> AIMessage:
    stop_reason = convert_stop_reason(message.stats.stop_reason)
    content = message.content

    result = AIMessage(content=content)
    result.response_metadata['stop_reason'] = stop_reason
    return result


def to_lms_chat(chat: Chat) -> lms.Chat:
    return lms.Chat.from_history(
        {'messages': [to_lms_message(message) for message in chat]}
    )


def from_lms_chat(chat: lms.Chat) -> Chat:
    return Chat(messages=[from_lms_message(message) for message in chat._get_history()])


def convert_stop_reason(
    stop_reason: lms._sdk_models.LlmPredictionStopReason,
) -> Literal['interrupt', 'stop', 'stop_token', 'length', 'tool_call']:
    finish_reason_mapping: dict[lms._sdk_models.LlmPredictionStopReason, str] = {
        'userStopped': 'interrupt',
        'modelUnloaded': 'interrupt',
        'failed': 'interrupt',
        'eosFound': 'stop',
        'stopStringFound': 'stop_token',
        'toolCalls': 'tool_call',
        'maxPredictedTokensReached': 'length',
        'contextLengthReached': 'length',
    }
    return finish_reason_mapping[stop_reason]


def tool_to_fun(tool: Tool):
    @wraps(tool.func)
    def wrapper(**kwargs):
        return tool.run(kwargs)

    return wrapper


def to_lms_fun_params(args: dict[str, Any]) -> dict[str, Any]:
    parameter_type_map: dict[str, type] = {
        'string': str,
        'number': float,
        'boolean': bool,
        'integer': int,
        'array': list,
        'object': object,
    }

    params = dict()
    for key in args:
        if not isinstance(args[key], dict):
            raise RuntimeError(
                f'Revice to_lms_fun_params function with argument {args}'
            )

        params[key] = parameter_type_map[args[key]['type']]
    return params


def to_lms_tools(tools: list[Tool]) -> list[lms.ToolFunctionDef]:
    return [
        lms.ToolFunctionDef(
            name=tool.name,
            description=tool.description,
            parameters=to_lms_fun_params(tool.args),
            implementation=tool_to_fun(tool),
        )
        for tool in tools
    ]
