from typing import Iterator, Literal, Optional

from pydantic import BaseModel

from radarange_orchestrator.formatting import ResponseFormat

from .backend import GenericModel
from .chat import Chat
from .config import BACKEND_CAPABILITIES
from .types.history import (
    AssistantMessage,
    AssistantMessageFragment,
    EmptyMessageHandler,
    MessageHandler,
)
from .types.tools import Tool

AVAILABLE_BACKEND = Literal['llama_cpp', 'lmstudio', 'local', 'remote']


class LLM_Config(BaseModel):
    ttl: int = 300
    gpus: list[int] = [0, 1]
    ctx_size: int = 80000


class Model:
    model_path: str
    backend: AVAILABLE_BACKEND
    config: LLM_Config
    model: GenericModel

    def __init__(
        self,
        model: str,
        backend: AVAILABLE_BACKEND = 'remote',
        config: Optional[LLM_Config] = None,
    ):
        self.model_path = model
        self.backend = backend
        match backend:
            case 'local':
                self.backend = 'llama_cpp'
            case 'remote':
                self.backend = 'lmstudio'

        self.config = config

        self.init_model()

    def init_model(self) -> None:
        match self.backend:
            case 'llama_cpp':
                if (
                    'llama_cpp' not in BACKEND_CAPABILITIES
                    or not BACKEND_CAPABILITIES['llama_cpp']['available']
                ):
                    raise NotImplementedError('llama_cpp backend is not enabled')

                from .backend import llama_cpp_model
                from .utils import find_model

                config = llama_cpp_model.to_llama_cpp_config(self.config)

                self.model = llama_cpp_model.LlamaModel(
                    find_model(self.model_path), config
                )
            case 'lmstudio':
                if (
                    'lmstudio' not in BACKEND_CAPABILITIES
                    or not BACKEND_CAPABILITIES['lmstudio']['available']
                ):
                    raise NotImplementedError('lmstudio backend is not enabled')

                from .backend import lmstudio_remote_model
                from .config import LMSTUDIO_ADDRESS, LMSTUDIO_PORT

                config = lmstudio_remote_model.to_lms_config(self.config)

                self.model = lmstudio_remote_model.LMSModel(
                    f'{LMSTUDIO_ADDRESS}:{LMSTUDIO_PORT}', self.model_path, config
                )

    def count_tokens(self, prompt: str | Chat) -> int:
        if isinstance(prompt, Chat):
            raise NotImplementedError('Counting tokens for chat is not yet implemented')

        return self.model.count_tokens(prompt)

    def close(self) -> None:
        if hasattr(self, 'model'):
            self.model.close()
            del self.model

    def create_chat_completion(
        self,
        chat: Chat,
        tools: list[Tool],
        response_format: Optional[ResponseFormat] = None,
        temperature: float = 0.7,
        max_tokens: int = 5000,
        stream: bool = False,
    ) -> AssistantMessage | Iterator[AssistantMessageFragment]:
        if not hasattr(self, 'model'):
            self.init_model()

        if response_format is not None and response_format.__repr__() != '':
            chat.add_system_message(f"""
            User wants you to answer in following format:
            {response_format.__repr__()}
            """)

        return self.model.create_chat_completion(
            chat, tools, response_format, temperature, max_tokens, stream
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
        if self.backend != 'lmstudio':
            raise NotImplementedError(
                f'Model.act is not implemented for {self.backend}'
            )

        return self.model.act(
            chat,
            tools,
            on_message,
            temperature,
            max_tokens_per_message,
            max_prediction_rounds,
        )

    @staticmethod
    def available_models(backend: AVAILABLE_BACKEND = 'remote') -> list[str]:
        if backend == 'llama_cpp' or backend == 'local':
            from utils import find_model

            return find_model('*')
        elif backend == 'lmstudio' or backend == 'remote':
            import lmstudio as lms

            from .config import LMSTUDIO_ADDRESS, LMSTUDIO_PORT

            with lms.Client(f'{LMSTUDIO_ADDRESS}:{LMSTUDIO_PORT}') as client:
                return [mod.model_key for mod in client.list_downloaded_models()]
        else:
            raise NotImplementedError(backend)
