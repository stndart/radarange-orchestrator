import atexit
import os
import weakref
from typing import Iterator, Literal, Optional

from llama_cpp import (
    LLAMA_SPLIT_MODE_LAYER,
    LLAMA_SPLIT_MODE_NONE,
    LLAMA_SPLIT_MODE_ROW,
    Llama,
)
from lmstudio import Chat
from pydantic import BaseModel

from radarange_orchestrator.formatting import ResponseFormat
from radarange_orchestrator.llm_backend import LLM_Config
from radarange_orchestrator.types.history import (
    AssistantMessage,
    AssistantMessageFragment,
)
from radarange_orchestrator.types.tools import Tool

from .generic_model import GenericModel


# Since Llama destructs ill while interpreter shutdown
@atexit.register
def _cleanup_all():
    for inst in list(_instances):
        try:
            inst.close()
        except Exception:
            pass


class LlamaConfig(BaseModel):
    gpus: list[int] = [0, 1]
    ctx_size: int = 0
    split_mode: Literal[
        LLAMA_SPLIT_MODE_NONE, LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_ROW  # type: ignore
    ]


def to_llama_cpp_config(config: LLM_Config) -> LlamaConfig:
    return LlamaConfig(
        gpus=config.gpus,
        ctx_size=config.ctx_size,
        split_mode=LLAMA_SPLIT_MODE_LAYER
    )


_instances = weakref.WeakSet()


class LlamaModel(GenericModel):
    model_path: str
    config: LlamaConfig
    llm: Llama

    def __init__(self, model_path: str, config: Optional[LlamaConfig] = None):
        if config is None:
            config = LlamaConfig()
        
        global _instances
        _instances.add(self)

        self.model_path = model_path
        self.config = config

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.config.gpus))

        # fit all the context and weights on gpu to perform 30 t/s
        if self.config.ctx_size <= 0:
            if len(self.config.gpus) == 1:
                self.config.ctx_size = int(10e3)
            elif len(self.config.gpus) == 2:
                self.config.ctx_size = int(80e3)

        tensor_split = [1.0]
        if len(self.config.gpus) == 2:
            tensor_split = [
                0.37176396722521365,
                0.3910968437984273,
            ]  # from lm_studio logs

        self.llm = Llama(
            seed=999,
            model_path=self.model_path,
            n_gpu_layers=-1,  # all
            split_mode=self.config.split_mode,  # the most optimal for speed
            main_gpu=1,  # ignored for layer split mode
            tensor_split=tensor_split,  # balancing layers between gpus
            n_ctx=self.config.ctx_size,
            n_batch=4096,  # todo: make tests
            n_threads=6,  # cpu threads
            flash_attn=True,  # Flash attention # don't know if works for QwQ:32b
            numa=3,  # Optimize NUMA allocation
            verbose=False,
        )
    
    def close(self) -> None:
        self.llm.close()
    
    def count_tokens(self, prompt: str | Chat):
        if isinstance(prompt, Chat):
            raise NotImplementedError('Counting tokens for chat is not yet implemented')
        
        return len(self.llm.tokenize(prompt.encode('utf-8')))

    def create_chat_completion(
        self,
        chat: Chat,
        tools: list[Tool],
        response_format: Optional[ResponseFormat] = None,
        temperature: float = 0.7,
        max_tokens: int = 5000,
        stream: bool = False,
    ) -> AssistantMessage | Iterator[AssistantMessageFragment]:
        grammar = response_format.grammar
        assert grammar is not None

        if stream:
            raise NotImplementedError('Stream mode is not yet implemented')
        else:
            # response: CreateChatCompletionResponse
            response = self.llm.create_chat_completion(
                chat,
                tools=tools,
                grammar=grammar,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            # TODO: add tool call parsing
            return AssistantMessage(
                role='assistant',
                content=response['choices'][0]['message']['content'],
                finish_reason=response['choices'][0]['finish_reason']
            )
