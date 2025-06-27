from functools import wraps
import json
from typing import Any, Callable, Iterator, Optional

import lmstudio as lms
from lmstudio._sdk_models import GpuSetting, LlmLoadModelConfig
from pydantic import BaseModel

from ..chat import Chat
from ..llm_backend import LLM_Config
from ..types.history import (
    AnyChatMessage,
    AssistantMessage,
    EmptyMessageHandler,
    FinishReason,
    MessageHandler,
    SystemPrompt,
    ToolCallResponse,
    UserMessage,
)
from ..types.tools import Tool, ToolHandler, ToolResult, parameter_type_map
from .generic_model import GenericModel


def convert_message(message: AnyChatMessage) -> lms.AnyChatMessage:
    if isinstance(message, SystemPrompt):
        return lms.SystemPrompt.from_dict(
            {
                'role': message.role,
                'content': [{'type': 'text', 'text': message.content}],
            }
        )
    elif isinstance(message, UserMessage):
        return lms.UserMessage.from_dict(
            {
                'role': message.role,
                'content': [{'type': 'text', 'text': message.content}],
            }
        )
    elif isinstance(message, AssistantMessage):
        return lms.AssistantResponse.from_dict(
            {
                'role': message.role,
                'content': [{'type': 'text', 'text': message.content}],
            }
        )
    elif isinstance(message, ToolCallResponse):
        return lms.ToolResultMessage.from_dict(
            {
                'role': message.role,
                'content': [
                    {
                        'type': 'toolCallResult',
                        'content': message.content,
                        'toolCallId': str(hash(message.content)),
                    }
                ],
            }
        )
    else:
        raise NotImplementedError(message.__class__)


def convert_chat(chat: Chat) -> lms.history.Chat:
    return lms.history.Chat.from_history(
        {'messages': [convert_message(message).to_dict() for message in chat]}
    )


def convert_finish_reason(
    finish_reason: lms._sdk_models.LlmPredictionStats,
) -> FinishReason:
    finish_reason_mapping: dict[lms._sdk_models.LlmPredictionStats, FinishReason] = {
        'userStopped': 'interrupt',
        'modelUnloaded': 'interrupt',
        'failed': 'interrupt',
        'eosFound': 'stop',
        'stopStringFound': 'stop_token',
        'toolCalls': 'tool_call',
        'maxPredictedTokensReached': 'length',
        'contextLengthReached': 'length',
    }
    return finish_reason_mapping[finish_reason]


def convert_assistant_message(message: lms.AssistantResponse) -> AssistantMessage:
    text = ''
    n_tools = 0
    for token in message.content:
        text += '\n'
        if token.type == 'text':
            text += token.text
        elif token.type == 'toolCallRequest':
            n_tools += 1
            text += json.dumps(token.tool_call_request.to_dict(), indent=2)

    return AssistantMessage(
        content=text, finish_reason='stop' if n_tools == 0 else 'tool_call'
    )


class LMSConfig(BaseModel):
    ttl: int = 300
    ctx_size: int = 80000


def to_lms_config(config: LLM_Config) -> LMSConfig:
    gpu_config = GpuSetting(disabled_gpus=list({0,1} ^ set(config.gpus)))
    return LMSConfig(gpu=gpu_config, ctx_size=config.ctx_size)


def tool_handler_to_impl(handler: ToolHandler) -> Callable[..., str]:
    @wraps(handler)
    def wrapper(*args, **kwargs) -> str:
        res: ToolResult = handler(*args, **kwargs)
        if res.status == 'success':
            return res.stdout
        else:
            return res.model_dump_json()

    return wrapper


class LMSModel(GenericModel):
    config: LMSConfig
    client: lms.Client
    model: lms.LLM

    def __init__(self, host: str, model: str, config: Optional[LMSConfig] = None):
        if config is None:
            config = LMSConfig()

        # TODO: add fields
        lms_config = lms.LlmLoadModelConfig(context_length=config.ctx_size)

        self.config = lms_config
        print(f'Connecting to host: {host}')
        self.client = lms.Client(host)
        self.model = self.client.llm.model(model, ttl=config.ttl, config=lms_config)

    def create_chat_completion(
        self,
        chat: Chat,
        tools: list[Tool],
        response_format: Optional[dict[str, str]] = None,
        grammar: Optional[Any] = None,  # TODO
        temperature: float = 0.7,
        max_tokens: int = 5000,
        stream: bool = False,
    ) -> AssistantMessage | Iterator[AssistantMessage]:
        history = convert_chat(chat)

        # TODO: add fields
        if stream:
            response: lms.PredictionStream = self.model.respond_stream(history=history)
            raise NotImplementedError('Stream support for lms is not implemented')
        else:
            response: lms.PredictionResult = self.model.respond(history=history)
            return AssistantMessage(
                content=response.content,
                finish_reason=convert_finish_reason(response.stats.stop_reason),
            )

    def act(
        self,
        chat: Chat,
        tools: list[Tool] = [],
        on_message: MessageHandler = EmptyMessageHandler,
        temperature: float = 0.7,
        max_tokens_per_message: int = -1,
        max_prediction_rounds: int = 3,
    ) -> AssistantMessage:
        assert max_prediction_rounds > 0

        history = convert_chat(chat)
        tool_defs = [
            lms.ToolFunctionDef(
                name=tool.definition.function.name,
                description=tool.definition.function.description,
                parameters={
                    prop: parameter_type_map[prop_value.type]
                    for prop, prop_value in tool.definition.function.parameters.properties.items()
                },
                implementation=tool_handler_to_impl(tool.handler),
            )
            for tool in tools
        ]

        def on_message_handler(message: lms.AssistantResponse | lms.ToolResultMessage):
            if isinstance(message, lms.AssistantResponse):
                normal_message = convert_assistant_message(message)
                history.append(message)
                on_message(normal_message)
            elif isinstance(message, lms.ToolResultMessage):
                for token in message.content:
                    normal_message = ToolCallResponse(
                        content=token.content, tool_call_id=token.tool_call_id
                    )
                    on_message(normal_message)
                history.append(message)
            else:
                raise NotImplementedError('Critical')

        self.model.act(
            history,
            tool_defs,
            on_message=on_message_handler,
            max_prediction_rounds=max_prediction_rounds,
            config=lms.LlmPredictionConfig(
                max_tokens=max_tokens_per_message, temperature=temperature
            ),
        )

        response: lms.AssistantResponse = history._get_last_message('assistant')

        return convert_assistant_message(response)
