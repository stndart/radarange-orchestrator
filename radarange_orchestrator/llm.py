import json
from typing import Any, Callable, Iterator, Optional

from .chat import Chat
from .config import DEFAULT_LLM_MODEL
from .llm_backend import AVAILABLE_BACKEND, LLM_Config, Model
from .formatting import ResponseFormat

# from .tools.grammar_switch_tool import GrammarContext, create_set_grammar_tool
from .types.history import (
    AssistantMessage,
    AssistantMessageFragment,
    EmptyMessageHandler,
    MessageHandler,
    SystemPrompt,
)
from .types.tools import Tool, ToolHandler, ToolRequest, ToolResult
from .utils.doc_to_json import make_tool_from_fun


class llm:
    """
    A high-level interface for interacting with a language model (LLM).

    Attributes:
        config: Configuration settings for the LLM.
        model: The underlying model instance initialized with specified parameters.
    """

    config: LLM_Config
    model: Model

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        backend: AVAILABLE_BACKEND = 'remote',
        config: LLM_Config = LLM_Config(),
    ):
        """
        Initialize the llm instance with a specific model, backend, and configuration.

        Args:
            model: Name of the language model to use (default is from DEFAULT_LLM_MODEL).
            backend: The execution environment for the model ('remote', 'local', etc.).
            config: LLM_Config object containing additional settings.
        """

        self.config = config
        self.model = Model(model, backend, config)
    
    def close(self) -> None:
        self.model.close()

    def chat(
        self, system_prompt: str = '', tools: list[Callable[..., Any]] | list[Tool] = []
    ) -> Chat:
        """
        Create a new chat session with an optional system prompt and available tools.

        Args:
            system_prompt: Initial instruction or context provided to the model.
            tools: List of functions or pre-configured tools accessible during the conversation.

        Returns:
            A Chat object initialized with the given system prompt and tools.
        """

        return Chat(SystemPrompt(content=system_prompt), tools=tools)
    
    def count_tokens(self, prompt: str | Chat) -> int:
        if isinstance(prompt, Chat):
            raise NotImplementedError('Counting tokens for chat is not yet implemented')
        
        return self.model.count_tokens(prompt)

    def respond_stream(
        self,
        prompt: Chat | str,
        tools: list[ToolHandler] | list[Tool] = [],
        temperature: float = 0.7,
        max_tokens: int = -1,
        response_format: Optional[ResponseFormat] = None,
    ) -> Iterator[AssistantMessageFragment]:
        """
        Generate a streaming response for the given prompt.

        Args:
            prompt: User input as a string or existing Chat object.
            tools: List of functions/tools available during response generation. They will be added to tools from chat.tools.
            temperature: Controls randomness in output (0.0-1.0, default=0.7).
            max_tokens: Maximum number of tokens to generate (-1 for no limit).
            response_format: Optional format constraint for the response.

        Returns:
            Iterator yielding AssistantMessageFragment chunks as they arrive.
        """

        chat: Chat = prompt if isinstance(prompt, Chat) else Chat()
        if not isinstance(prompt, Chat):
            chat.add_user_message(prompt)

        chat_completion = self.model.create_chat_completion(
            chat,
            tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        return chat_completion  # TODO

    def respond(
        self,
        prompt: Chat | str,
        tools: list[ToolHandler] | list[Tool] = [],
        temperature: float = 0.7,
        max_tokens: int = -1,
        response_format: Optional[ResponseFormat] = None,
    ) -> AssistantMessage:
        """
        Generate a complete non-streaming response for the given prompt.

        Args:
            prompt: User input as a string or existing Chat object.
            tools: List of functions/tools available during response generation. They will be added to tools from chat.tools.
            temperature: Controls randomness in output (0.0-1.0, default=0.7).
            max_tokens: Maximum number of tokens to generate (-1 for no limit).
            response_format: Optional format constraint for the response.

        Returns:
            A complete AssistantMessage containing the generated content.
        """

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
        """
        Execute a series of tool calls against the provided functions/tools.

        Args:
            tool_calls: List of ToolRequest objects specifying which tools to call.
            tools: Available tools that can be executed.

        Returns:
            List of ToolResult objects containing execution outcomes, including errors.
        """

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
        response_format: Optional[ResponseFormat] = None,  # BETA
    ) -> AssistantMessage:
        """
        Execute a multi-turn interaction where the model generates responses and potentially calls tools.

        For local/llama_cpp backends, performs up to max_prediction_rounds of message/tool call cycles.
        Remote/lmstudio delegates directly to model.act. Raises error for unsupported backends.

        Args:
            prompt: Starting point as Chat or string (will be accounted as user message).
            tools: Functions/tools available for execution in this context.
            on_message: Callback handler triggered when new messages are generated.
            temperature: Controls response randomness (0.0-1.0, default=0.7).
            max_tokens_per_message: Maximum tokens per message generation (-1 for no limit).
            max_prediction_rounds: Max number of reasoning/tool call cycles to perform.

        Returns:
            The final AssistantMessage from the interaction sequence.
        """

        chat = Chat(prompt) if isinstance(prompt, str) else prompt.copy()
        tools = [
            fun if isinstance(fun, Tool) else make_tool_from_fun(fun) for fun in tools
        ]

        if response_format is not None and response_format.__repr__() != '':
            chat.add_system_message(f"""
            User wants you to answer in the following format:
            {response_format.__repr__()}
            """)

        assert max_prediction_rounds > 0
        if self.model.backend == 'llama_cpp':
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
        elif self.model.backend == 'lmstudio':
            response = self.model.act(
                chat,
                tools + chat.tools,
                on_message,
                temperature,
                max_tokens_per_message,
                max_prediction_rounds,
            )
        else:
            raise NotImplementedError(
                f'llm.act is not implemented for {self.model.backend}'
            )

        return response
