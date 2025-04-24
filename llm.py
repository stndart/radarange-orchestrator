import atexit
import json
import os
import weakref
from typing import Any, Callable, Iterator, Optional

from llama_cpp import LLAMA_SPLIT_MODE_LAYER, JsonType, Llama, LlamaGrammar

from .chat import Chat
from .llama_cpp_bindings import from_chat_completion, to_llama_tools
from .llm_types import (
    ChatMessage,
    EmptyMessageHandler,
    LlmConfig,
    MessageHandler,
    Response,
    ResponseStream,
)
from .tools.tool_annotation import (
    Tool,
    ToolCall,
    ToolHandler,
    ToolResult,
    tool_result_to_str,
)
from .utils import find_model, make_tool_from_fun
from .tools.grammar_switch_tool import GrammarContext, create_set_grammar_tool

_instances = weakref.WeakSet()


# Since Llama destructs ill while interpreter shutdown
@atexit.register
def _cleanup_all():
    for inst in list(_instances):
        try:
            inst.close()
        except Exception:
            pass


class llm:
    def __init__(
        self,
        model: str = "QwQ-32B-Q4_K_M.gguf",
        config: LlmConfig = LlmConfig(gpus=[0, 1], split_mode=LLAMA_SPLIT_MODE_LAYER),
    ):
        global _instances
        _instances.add(self)

        self.model_path = find_model(model)
        self.config = config

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.gpus))

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

    def close(self):
        if hasattr(self, "llm"):
            del self.llm

    def chat(
        self, system_prompt: str = "", tools: list[Callable[..., Any]] | list[Tool] = []
    ) -> Chat:
        return Chat(ChatMessage(role="system", content=system_prompt), tools=tools)

    def respond_stream(
        self, prompt: Chat, temperature: float = 0.7, max_tokens: int = -1
    ) -> ResponseStream:
        raise NotImplementedError("Stream support is not yet implemented")

    def respond(
        self,
        prompt: Chat | str,
        tools: list[ToolHandler] | list[Tool] = [],
        temperature: float = 0.7,
        max_tokens: int = -1,
        response_format: Optional[str] = None,
        grammar: Optional[LlamaGrammar] = None,
    ) -> Response:
        if isinstance(prompt, str):
            return self.respond(
                Chat(prompt),
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        else:
            full_tools: list[Tool] = prompt.tools + [
                fun if isinstance(fun, Tool) else make_tool_from_fun(fun)
                for fun in tools
            ]
            chat_completion = self.llm.create_chat_completion(
                messages=prompt.llama_messages(),
                tools=to_llama_tools(full_tools),
                response_format=None
                if not response_format
                else {"type": "json_object", "schema": response_format},
                grammar=grammar,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            self.last_response = chat_completion
            assert not isinstance(
                chat_completion, Iterator
            )  # should not fail since stream=False
            return from_chat_completion(chat_completion)

    def invoke_tool_calls(
        self, tool_calls: list[ToolCall], tools: list[Tool]
    ) -> list[ToolResult]:
        def find_tool(name: str) -> Optional[Tool]:
            for t in tools:
                if t.definition.function.name == name:
                    return t

        res: list[ToolResult] = []
        for call in tool_calls:
            tool = find_tool(call.function.name)
            if not tool:
                res.append(
                    ToolResult(
                        status="error",
                        stdout="",
                        stderr="Tool call format error",
                        returncode=-1,
                    )
                )
            else:
                try:
                    res.append(tool.handler(**json.loads(call.function.arguments)))
                except Exception as e:
                    res.append(
                        ToolResult(
                            status="error",
                            stdout="",
                            stderr=f"Tool call error: {e}\nTraceback: {e.__traceback__}",
                            returncode=-1,
                        )
                    )
        return res

    def act(
        self,
        prompt: Chat | str,
        tools: list[ToolHandler] | list[Tool] = [],
        on_message: MessageHandler = EmptyMessageHandler,
        temperature: float = 0.7,
        max_tokens_per_message: int = -1,
        max_prediction_rounds: int = 3,
    ) -> Response:
        chat = Chat(prompt) if isinstance(prompt, str) else prompt.copy()
        tools = [
            fun if isinstance(fun, Tool) else make_tool_from_fun(fun) for fun in tools
        ]

        assert max_prediction_rounds > 0
        for i in range(max_prediction_rounds):
            response = self.respond(
                chat,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens_per_message,
            )
            on_message(response)
            chat.append(response)
            if response.finish_reason == "tool_call":
                results: list[ToolResult] = self.invoke_tool_calls(
                    response.tool_calls, chat.tools + tools
                )
                for call, res in zip(response.tool_calls, results):
                    chat.add_tool_message(tool_result_to_str(res), call.id)
                    on_message(chat[-1])
            else:
                break
        return response

    def act_with_weak_json(
        self,
        prompt: Chat | str,
        json_format: Optional[JsonType] = None,
        tools: list[ToolHandler] | list[Tool] = [],
        on_message: MessageHandler = EmptyMessageHandler,
        temperature: float = 0.7,
        max_tokens_per_message: int = -1,
        max_prediction_rounds: int = 3,
    ) -> tuple[Response, Chat]:
        chat = Chat(prompt) if isinstance(prompt, str) else prompt.copy()
        tools = [
            fun if isinstance(fun, Tool) else make_tool_from_fun(fun) for fun in tools
        ]

        if json_format is not None:
            pre_prompt = f"""
User wants you to answer following format:
{json.dumps(json_format, indent=2)}
"""
            chat.add_system_message(pre_prompt)

        assert max_prediction_rounds > 0
        for i in range(max_prediction_rounds):
            response = self.respond(
                chat,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens_per_message,
            )
            on_message(response)
            chat.append(response)
            if response.finish_reason == "tool_call":
                results: list[ToolResult] = self.invoke_tool_calls(
                    response.tool_calls, chat.tools + tools
                )
                for call, res in zip(response.tool_calls, results):
                    chat.add_tool_message(tool_result_to_str(res), call.id)
                    on_message(chat[-1])
            else:
                break
        return response, chat

    DEFAULT_GRAMMAR_SYSTEM_PROMPT = """
User expects you to fulfill the task in a few separate steps.
Since output is strictly formatted, but perhaps you need to make some intermediate generations, grammar is yet unset.
Before telling the answer to user, call grammar switch tool and wait for it to successfully complete.
The next response from you will be applied with correct grammar, so make sure to separate output into separate message.
"""

    # WIP
    def act_with_grammar(
        self,
        prompt: Chat | str,
        tools: list[Callable[..., Any]] | list[Tool] = [],
        on_message: MessageHandler = EmptyMessageHandler,
        temperature: float = 0.7,
        max_tokens_per_message: int = -1,
        max_prediction_rounds: int = 3,
        response_format: Optional[str] = None,  # beta
        grammar: Optional[LlamaGrammar] = None,  # beta
        system_prompt: str = DEFAULT_GRAMMAR_SYSTEM_PROMPT,  # beta
        grammar_context: Optional[GrammarContext] = None,  # beta
    ) -> tuple[Response, Chat]:
        assert not (response_format is not None and grammar is not None), (
            "Only one of response_format or grammar can be set"
        )

        chat = Chat(prompt) if isinstance(prompt, str) else prompt.copy()
        full_tools: list[Tool] = [
            fun if isinstance(fun, Tool) else make_tool_from_fun(fun) for fun in tools
        ]

        if grammar_context is None:
            grammar_context, _ = create_set_grammar_tool()

            if response_format:
                grammar = LlamaGrammar.from_json_schema(response_format)
            if grammar:
                grammar_context, fly_tool = create_set_grammar_tool(grammar)
                chat.add_system_message(system_prompt)
                full_tools.append(fly_tool)

            chat.tools += full_tools

        assert max_prediction_rounds > 0
        for i in range(max_prediction_rounds):
            response = self.respond(
                chat,
                tools=full_tools,
                temperature=temperature,
                max_tokens=max_tokens_per_message,
                response_format=response_format,
                grammar=grammar_context.get_current_grammar(),
            )
            on_message(response)
            chat.append(response)
            if response.finish_reason == "tool_call":
                results: list[ToolResult] = self.invoke_tool_calls(
                    response.tool_calls, chat.tools + full_tools
                )
                for call, res in zip(response.tool_calls, results):
                    chat.add_tool_message(tool_result_to_str(res), call.id)
                    on_message(chat[-1])
            else:
                break
        return response, chat
