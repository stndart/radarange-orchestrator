from typing import Iterator, Optional

from .chat import (
    Chat,
    ToolMessage,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    MessageHandler,
    EmptyMessageHandler,
)
from .config import DEFAULT_LLM_MODEL
from .formatting import ResponseFormat
from .llm_backend import AVAILABLE_BACKEND, LLM_Config, Model
from .tools import Tool, ToolCall, InvalidToolCall


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
            backend: The execution environment for the model ('remote', 'local', etc.). See available backends in llm_backend.AVAILABLE_BACKEND.
            config: LLM_Config object containing additional settings.
        """

        self.config = config
        self.model = Model(model, backend, config)

    def close(self) -> None:
        self.model.close()

    def chat(self, system_prompt: str = '', tools: list[Tool] = []) -> Chat:
        """
        Create a new chat session with an optional system prompt and available tools.

        Args:
            system_prompt: Initial instruction or context provided to the model.
            tools: List of functions or pre-configured tools accessible during the conversation.

        Returns:
            A Chat object initialized with the given system prompt and tools.
        """

        chat = Chat(tools=tools)
        chat.add_message(SystemMessage(content=system_prompt))
        return chat

    def count_tokens(self, prompt: str | Chat) -> int:
        return self.model.count_tokens(prompt)

    def respond_stream(
        self,
        prompt: Chat | str,
        tools: list[Tool] = [],
        temperature: float = 0.7,
        max_tokens: int = -1,
        response_format: Optional[ResponseFormat] = None,
    ) -> Iterator[AIMessageChunk]:
        """
        Generate a streaming response for the given prompt.

        Args:
            prompt: User input as a string or existing Chat object.
            tools: List of langchain StructuredTool available during response generation. They will be added to chat.tools.
            temperature: Controls randomness in output (0.0-1.0, default=0.7).
            max_tokens: Maximum number of tokens to generate (-1 for no limit).
            response_format: Optional format constraint for the response.

        Returns:
            Iterator yielding AIMessageChunk chunks as they arrive.
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
        tools: list[Tool] = [],
        temperature: float = 0.7,
        max_tokens: int = -1,
        response_format: Optional[ResponseFormat] = None,
    ) -> AIMessage:
        """
        Generate a complete non-streaming response for the given prompt.

        Args:
            prompt: User input as a string or existing Chat object.
            tools: List of langchain StructuredTool available during response generation. They will be added to chat.tools.
            temperature: Controls randomness in output (0.0-1.0, default=0.7).
            max_tokens: Maximum number of tokens to generate (-1 for no limit).
            response_format: Optional format constraint for the response.

        Returns:
            A complete AIMessage containing the generated content.
        """

        chat: Chat
        if isinstance(prompt, str):
            chat = Chat()
            chat.add_user_message(prompt)
        else:
            chat = prompt

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
        self, tool_calls: list[ToolCall | InvalidToolCall], tools: list[Tool]
    ) -> list[ToolMessage]:
        """
        Execute a series of tool calls against the provided functions/tools.

        Args:
            tool_calls: List of ToolCall objects specifying which tools to call.
            tools: Available tools that can be executed.

        Returns:
            List of ToolMessage objects containing execution outcomes, including errors.
        """

        def find_tool(name: str) -> Optional[Tool]:
            for t in tools:
                if t.name == name:
                    return t

        res: list[ToolMessage] = []
        for call in tool_calls:
            tool = find_tool(call['name']) if call['type'] == 'tool_call' else None
            if not tool:
                res.append(
                    ToolMessage(
                        'Tool call format error',
                        tool_call_id='invalid_tool_call',
                        status='error',
                    )
                )
            else:
                try:
                    res.append(
                        tool.run(tool_input=call['args'], tool_call_id=call['id'])
                    )
                except Exception as e:
                    res.append(
                        ToolMessage(
                            f'Tool call error: {e}\nTraceback: {e.__traceback__}',
                            tool_call_id='invalid_tool_call',
                            status='error',
                        )
                    )
        return res

    def act(
        self,
        prompt: Chat | str,
        tools: list[Tool] = [],
        on_message: MessageHandler = EmptyMessageHandler,
        temperature: float = 0.7,
        max_tokens_per_message: int = -1,
        max_prediction_rounds: int = 3,
        response_format: Optional[ResponseFormat] = None,  # BETA
    ) -> AIMessage:
        """
        Execute a multi-turn interaction where the model generates responses and potentially calls tools.

        For local/llama_cpp backends, performs up to max_prediction_rounds of message/tool call cycles.
        Remote/lmstudio delegates directly to model.act. Raises error for unsupported backends.

        Args:
            prompt: Starting point as Chat or string (will be accounted as user message).
            tools: langchain StructuredTool tools available for execution in this context.
            on_message: Callback handler triggered when new messages are generated.
            temperature: Controls response randomness (0.0-1.0, default=0.7).
            max_tokens_per_message: Maximum tokens per message generation (-1 for no limit).
            max_prediction_rounds: Max number of reasoning/tool call cycles to perform.

        Returns:
            The final AIMessage from the interaction sequence.
        """
        
        chat: Chat
        if isinstance(prompt, str):
            chat = Chat()
            chat.add_user_message(prompt)
        else:
            chat = prompt.model_copy(deep=True)

        if response_format is not None and response_format.__repr__() != '':
            chat.add_message(
                SystemMessage(
                    content=f"""
User wants you to answer in the following format:
{response_format.__repr__()}"""
                )
            )

        assert max_prediction_rounds > 0
        if self.model.backend == 'llama_cpp':
            for i in range(max_prediction_rounds):
                response: AIMessage = self.respond(
                    chat,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens_per_message,
                )
                on_message(response)
                chat.add_message(response)
                if response.response_metadata.get('stop_reason', 'stop') == 'tool_call':
                    results: list[ToolMessage] = self.invoke_tool_calls(
                        response.tool_calls + response.invalid_tool_calls,
                        chat.tools + tools,
                    )
                    chat.add_messages(results)
                    for message in results:
                        on_message(message)
                else:
                    break
        elif self.model.backend == 'lmstudio':
            response = self.model.act(
                chat,
                tools,
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
