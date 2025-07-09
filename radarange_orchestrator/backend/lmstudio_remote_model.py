from typing import Iterator, Optional

import lmstudio as lms
from pydantic import BaseModel
from lmstudio.sync_api import SyncSessionLlm

from ..chat import (
    AIMessage,
    AIMessageChunk,
    Chat,
    EmptyMessageHandler,
    MessageHandler,
)
from ..formatting import ResponseFormat
from ..llm_backend import LLM_Config
from ..tools import Tool
from .generic_model import GenericModel
from .lmstudio_bindings import from_lms_message, from_lms_response, to_lms_chat, to_lms_tools


class Gpu(BaseModel):
    ratio: float = 1.0
    mainGpu: int = 0
    disabledGpus: list[int] = []


class LMSConfig(BaseModel):
    ttl: int = 300
    gpu: Gpu
    ctx_size: int = 80000


def to_lms_config(config: LLM_Config) -> LMSConfig:
    gpu_config = Gpu(disabledGpus=list({0, 1} ^ set(config.gpus)))
    return LMSConfig(gpu=gpu_config, ctx_size=config.ctx_size, ttl=config.ttl)


class LMSModel(GenericModel):
    model_id: str
    default_ttl: int
    config: LMSConfig
    client: lms.Client
    model: lms.LLM

    def __init__(self, host: str, model: str, config: Optional[LMSConfig] = None):
        if config is None:
            config = LMSConfig()

        # TODO: add fields
        lms_config = lms.LlmLoadModelConfig(
            gpu=config.gpu.model_dump(), context_length=config.ctx_size
        )

        self.model_id = model
        self.default_ttl = config.ttl
        self.config = lms_config

        print(f'Connecting to host: {host}')
        self.client = lms.Client(host)
        self.model = self.client.llm.model(self.model_id, ttl=self.default_ttl, config=self.config)

    def close(self) -> None:
        self.model.unload()

    def count_tokens(self, prompt: str | Chat):
        if isinstance(prompt, Chat):
            raise NotImplementedError('Counting tokens for chat is not yet implemented')

        return self.model.count_tokens(prompt)
    
    def assure_loaded(self) -> None:
        loaded = [m.identifier for m in self.client._get_session(SyncSessionLlm).list_loaded()]
        if self.model.identifier not in loaded:
            self.model = self.client.llm.model(self.model_id, ttl=self.default_ttl, config=self.config)
        

    def create_chat_completion(
        self,
        chat: Chat,
        tools: list[Tool],
        response_format: Optional[ResponseFormat] = None,
        temperature: float = 0.7,
        max_tokens: int = 5000,
        stream: bool = False,
    ) -> AIMessage | Iterator[AIMessageChunk]:
        all_tools = tools + chat.tools
        all_tools = to_lms_tools(all_tools)

        # TODO: add fields
        if stream:
            response: lms.PredictionStream = self.model.respond_stream(
                history=to_lms_chat(chat)
            )
            raise NotImplementedError('Stream support for lms is not implemented')
        else:
            # TODO: add tools
            if len(all_tools) > 0:
                raise NotImplementedError('Tool support for lms.llm.respond is not implemented')

            response: lms.PredictionResult = self.model.respond(
                history=to_lms_chat(chat),
                response_format=response_format.json_schema,
                config={
                    'temperature': temperature,
                    'maxTokens': max_tokens if max_tokens > 0 else None,
                },
            )
            return from_lms_response(response)

    def act(
        self,
        chat: Chat,
        tools: list[Tool] = [],
        on_message: MessageHandler = EmptyMessageHandler,
        temperature: float = 0.7,
        max_tokens_per_message: int = -1,
        max_prediction_rounds: int = 3,
    ) -> AIMessage:
        assert max_prediction_rounds > 0
        
        all_tools = tools + chat.tools
        all_tools = to_lms_tools(all_tools)

        def on_message_handler(message: lms.AssistantResponse | lms.ToolResultMessage):
            if isinstance(message, lms.AssistantResponse):
                normal_message: AIMessage = from_lms_message(message)
                on_message(normal_message)
                if on_message != chat.add_message:
                    chat.add_message(normal_message)
            elif isinstance(message, lms.ToolResultMessage):
                for normal_message in from_lms_message(message):
                    on_message(normal_message)
                    if on_message != chat.add_message:
                        chat.add_message(normal_message)
            else:
                raise NotImplementedError('Critical error: on_message handler received something not assistant response or tool call response')

        if max_tokens_per_message == -1:
            max_tokens_per_message = None
        
        self.model.act(
            chat=to_lms_chat(chat),
            tools=all_tools,
            on_message=on_message_handler,
            max_prediction_rounds=max_prediction_rounds,
            config=lms.LlmPredictionConfig(
                max_tokens=max_tokens_per_message, temperature=temperature
            ),
        )

        return chat[-1]
