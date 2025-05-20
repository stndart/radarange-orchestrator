import json
from typing import Any, Callable, Iterator, Optional

from .chat import Chat
from .config import DEFAULT_LLM_MODEL
from .llm_backend import AVAILABLE_BACKEND, LLM_Config, Model

# from .tools.grammar_switch_tool import GrammarContext, create_set_grammar_tool
from .types.history import (
    AssistantMessage,
    AssistantMessageFragment,
    EmptyMessageHandler,
    MessageHandler,
    SystemPrompt,
    UserMessage,
)
from .types.tools import Tool, ToolHandler, ToolRequest, ToolResult
from .utils.doc_to_json import make_tool_from_fun


class llm:
    config: LLM_Config
    model: Model

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        backend: AVAILABLE_BACKEND = 'remote',
        config: LLM_Config = LLM_Config(),
    ):
        self.config = config
        self.model = Model(model, backend, config)

    def chat(
        self, system_prompt: str = '', tools: list[Callable[..., Any]] | list[Tool] = []
    ) -> Chat:
        return Chat(SystemPrompt(content=system_prompt), tools=tools)

    def respond_stream(
        self,
        prompt: Chat | str,
        tools: list[ToolHandler] | list[Tool] = [],
        temperature: float = 0.7,
        max_tokens: int = -1,
        response_format: Optional[str] = None,
    ) -> Iterator[AssistantMessageFragment]:
        chat = prompt if isinstance(prompt, Chat) else UserMessage(prompt)

        chat_completion = self.model.create_chat_completion(
            chat,
            tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        return chat_completion

    def respond(
        self,
        prompt: Chat | str,
        tools: list[ToolHandler] | list[Tool] = [],
        temperature: float = 0.7,
        max_tokens: int = -1,
        response_format: Optional[str] = None,
    ) -> AssistantMessage:
        chat: Chat = prompt if isinstance(prompt, Chat) else Chat()
        if not isinstance(prompt, Chat):
            chat.add_user_message(prompt)

        chat_completion = self.model.create_chat_completion(
            chat,
            tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return chat_completion

    def invoke_tool_calls(
        self, tool_calls: list[ToolRequest], tools: list[Tool]
    ) -> list[ToolResult]:
        def find_tool(name: str) -> Optional[Tool]:
            for t in tools:
                if t.definition.function.name == name:
                    return t

        res: list[ToolResult] = []
        for call in tool_calls:
            tool = find_tool(call.name)
            if not tool:
                res.append(
                    ToolResult(
                        status='error',
                        stdout='',
                        stderr='Tool call format error',
                        returncode=-1,
                    )
                )
            else:
                try:
                    res.append(tool.handler(**json.loads(call.arguments)))
                except Exception as e:
                    res.append(
                        ToolResult(
                            status='error',
                            stdout='',
                            stderr=f'Tool call error: {e}\nTraceback: {e.__traceback__}',
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
    ) -> AssistantMessage:
        chat = Chat(prompt) if isinstance(prompt, str) else prompt.copy()
        tools = [
            fun if isinstance(fun, Tool) else make_tool_from_fun(fun) for fun in tools
        ]

        assert max_prediction_rounds > 0
        for i in range(max_prediction_rounds):
            response: AssistantMessage = self.respond(
                chat,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens_per_message,
            )
            on_message(response)
            chat.append(response)
            if response.finish_reason == 'tool_call':
                results: list[ToolResult] = self.invoke_tool_calls(
                    response.tool_calls, chat.tools + tools
                )
                for call, res in zip(response.tool_calls, results):
                    chat.add_tool_message(res.model_dump_json(), call.id)
                    on_message(chat[-1])
            else:
                break
        return response


#     def act_with_weak_json(
#         self,
#         prompt: Chat | str,
#         json_format: Optional[JsonType] = None,
#         tools: list[ToolHandler] | list[Tool] = [],
#         on_message: MessageHandler = EmptyMessageHandler,
#         temperature: float = 0.7,
#         max_tokens_per_message: int = -1,
#         max_prediction_rounds: int = 3,
#     ) -> tuple[Response, Chat]:
#         chat = Chat(prompt) if isinstance(prompt, str) else prompt.copy()
#         tools = [
#             fun if isinstance(fun, Tool) else make_tool_from_fun(fun) for fun in tools
#         ]

#         if json_format is not None:
#             pre_prompt = f"""
# User wants you to answer following format:
# {json.dumps(json_format, indent=2)}
# """
#             chat.add_system_message(pre_prompt)

#         assert max_prediction_rounds > 0
#         for i in range(max_prediction_rounds):
#             response = self.respond(
#                 chat,
#                 tools=tools,
#                 temperature=temperature,
#                 max_tokens=max_tokens_per_message,
#             )
#             on_message(response)
#             chat.append(response)
#             if response.finish_reason == 'tool_call':
#                 results: list[ToolResult] = self.invoke_tool_calls(
#                     response.tool_calls, chat.tools + tools
#                 )
#                 for call, res in zip(response.tool_calls, results):
#                     chat.add_tool_message(tool_result_to_str(res), call.id)
#                     on_message(chat[-1])
#             else:
#                 break
#         return response, chat

#     DEFAULT_GRAMMAR_SYSTEM_PROMPT = """
# User expects you to fulfill the task in a few separate steps.
# Since output is strictly formatted, but perhaps you need to make some intermediate generations, grammar is yet unset.
# Before telling the answer to user, call grammar switch tool and wait for it to successfully complete.
# The next response from you will be applied with correct grammar, so make sure to separate output into separate message.
# """

#     # WIP
#     def act_with_grammar(
#         self,
#         prompt: Chat | str,
#         tools: list[Callable[..., Any]] | list[Tool] = [],
#         on_message: MessageHandler = EmptyMessageHandler,
#         temperature: float = 0.7,
#         max_tokens_per_message: int = -1,
#         max_prediction_rounds: int = 3,
#         response_format: Optional[str] = None,  # beta
#         grammar: Optional[LlamaGrammar] = None,  # beta
#         system_prompt: str = DEFAULT_GRAMMAR_SYSTEM_PROMPT,  # beta
#         grammar_context: Optional[GrammarContext] = None,  # beta
#     ) -> tuple[Response, Chat]:
#         assert not (response_format is not None and grammar is not None), (
#             'Only one of response_format or grammar can be set'
#         )

#         chat = Chat(prompt) if isinstance(prompt, str) else prompt.copy()
#         full_tools: list[Tool] = [
#             fun if isinstance(fun, Tool) else make_tool_from_fun(fun) for fun in tools
#         ]

#         if grammar_context is None:
#             grammar_context, _ = create_set_grammar_tool()

#             if response_format:
#                 grammar = LlamaGrammar.from_json_schema(response_format)
#             if grammar:
#                 grammar_context, fly_tool = create_set_grammar_tool(grammar)
#                 chat.add_system_message(system_prompt)
#                 full_tools.append(fly_tool)

#             chat.tools += full_tools

#         assert max_prediction_rounds > 0
#         for i in range(max_prediction_rounds):
#             response = self.respond(
#                 chat,
#                 tools=full_tools,
#                 temperature=temperature,
#                 max_tokens=max_tokens_per_message,
#                 response_format=response_format,
#                 grammar=grammar_context.get_current_grammar(),
#             )
#             on_message(response)
#             chat.append(response)
#             if response.finish_reason == 'tool_call':
#                 results: list[ToolResult] = self.invoke_tool_calls(
#                     response.tool_calls, chat.tools + full_tools
#                 )
#                 for call, res in zip(response.tool_calls, results):
#                     chat.add_tool_message(tool_result_to_str(res), call.id)
#                     on_message(chat[-1])
#             else:
#                 break
#         return response, chat
