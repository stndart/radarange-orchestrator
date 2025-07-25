from typing import Iterator, Literal, Optional

from pydantic import BaseModel

from .backend import GenericModel
from .chat import (
    AIMessage,
    AIMessageChunk,
    Chat,
    EmptyMessageHandler,
    MessageHandler,
    SystemMessage,
)
from .config import BACKEND_CAPABILITIES
from .formatting import ResponseFormat
from .tools import Tool

AVAILABLE_BACKEND = Literal['llama_cpp', 'lmstudio', 'local', 'remote']
DEFAULT_LOCAL_BACKEND = 'llama_cpp'
DEFAULT_REMOTE_BACKEND = 'lmstudio'


class LLM_Config(BaseModel):
    ttl: int = 300
    gpus: list[int] = [0, 1]
    ctx_size: int = 80000


class Model:
    """
    High-level wrapper for LLMs in different backends
    Delegates various methods such as create_chat_completion() and act() to appropriate backends
    """

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
                self.backend = DEFAULT_LOCAL_BACKEND
            case 'remote':
                self.backend = DEFAULT_REMOTE_BACKEND

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

                model = find_model(self.model_path)
                if not isinstance(model, str):
                    print(
                        f'Warning: multiple models are available by {self.model_path}. Taking the first from the list: {model}.'
                    )
                    assert len(model) > 0
                    model = model[0]

                self.model = llama_cpp_model.LlamaModel(model, config)
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
    ) -> AIMessage | Iterator[AIMessageChunk]:
        if not hasattr(self, 'model'):
            self.init_model()

        self.model.assure_loaded()

        if response_format is not None and response_format.__repr__() != '':
            chat.add_message(
                SystemMessage(f"""
            User wants you to answer in the following format:
            {response_format.__repr__()}
            """)
            )

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
    ) -> AIMessage:
        if not hasattr(self, 'model'):
            self.init_model()

        self.model.assure_loaded()

        # if self.backend != 'lmstudio':
        if not hasattr(self.model, 'act'):
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
            from .utils import find_model

            models = find_model('*')
            if isinstance(models, str):
                models = [models]
            return models
        elif backend == 'lmstudio' or backend == 'remote':
            import lmstudio as lms

            from .config import LMSTUDIO_ADDRESS, LMSTUDIO_PORT

            with lms.Client(f'{LMSTUDIO_ADDRESS}:{LMSTUDIO_PORT}') as client:
                return [mod.model_key for mod in client.list_downloaded_models()]
        else:
            raise NotImplementedError(backend)
