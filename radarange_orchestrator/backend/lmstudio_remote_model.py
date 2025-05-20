from typing import Any, Iterator, Optional

import lmstudio as lms
from pydantic import BaseModel

from ..chat import Chat
from ..llm_backend import LLM_Config
from ..types.history import AssistantMessage
from ..types.tools import Tool
from .generic_model import GenericModel


def convert_chat(chat: Chat) -> lms.history.Chat:
    return lms.history.Chat.from_history(
        {
            'messages': [
                {'role': message.role, 'content': message.content} for message in chat
            ]
        }
    )


class LMSConfig(BaseModel):
    ttl: int = 300
    ctx_size: int = 80000


def to_lms_config(config: LLM_Config) -> LMSConfig:
    return LMSConfig(ctx_size=config.ctx_size)


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
            print(history)
            response: lms.PredictionResult = self.model.respond(history=history)
            return AssistantMessage(content=response.content)
