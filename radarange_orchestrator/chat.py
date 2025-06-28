from __future__ import annotations

import os
from typing import Iterator

from IPython.display import Markdown, display

from .types.history import (
    AnyChatMessage,
    AssistantMessage,
    SystemPrompt,
    ToolCallResponse,
    UserMessage,
)
from .types.tools import Tool, ToolHandler
from .utils import display_message, is_list_of, make_tool_from_fun
from .utils.extract_tool_calls import remove_think_block


class Chat:
    """
    A container class for managing chat history and tool configurations during interactions with an LLM.

    Attributes:
        history: List of chat messages in the conversation (UserMessage, SystemPrompt, etc.).
        tools: Available functions/tools that can be called by the model.
    """

    history: list[AnyChatMessage]
    tools: list[Tool]

    def __init__(
        self,
        prompt: str | AnyChatMessage = '',
        tools: list[ToolHandler] | list[Tool] = [],
    ):
        """
        Initialize a Chat instance with an optional initial message and available tools.

        Args:
            prompt: Starting message as string (converted to UserMessage) or pre-defined message.
            tools: List of functions/tool handlers available in this chat context.
                   If not already Tool instances, they will be wrapped using make_tool_from_fun.
        """

        self.history = []
        self.tools = []

        if isinstance(prompt, str):
            if prompt != '':
                self.add_user_message(prompt)
        else:
            if prompt.content != '':
                self.append(prompt)

        if len(tools) > 0:
            if is_list_of(tools, Tool):
                self.tools = tools
            else:
                self.tools = [make_tool_from_fun(fun) for fun in tools]  # type: ignore

    def copy(self) -> Chat:
        """
        Create a deep copy of this Chat instance.

        Returns:
            New Chat object with identical message history and tool configuration.
        """

        new_chat = Chat()
        new_chat.tools = self.tools  # no need to copy since tools are const
        for i, message in enumerate(self.history):
            new_chat.append(message.model_copy(deep=True))

        return new_chat

    def append(self, message: AnyChatMessage):
        """
        Add a message to the chat history.

        Args:
            message: Message object to be appended.
        """

        self.history.append(message)

    def add_user_message(self, prompt: str) -> UserMessage:
        """
        Create and add a user message to the conversation.

        Args:
            prompt: Text content of the user's input.

        Returns:
            Created UserMessage instance added to history.
        """

        message = UserMessage(content=prompt)
        self.append(message)
        return message

    def add_system_message(self, prompt: str) -> SystemPrompt:
        """
        Create and add a system instruction/prompt to the conversation context.

        Args:
            prompt: Text content of the system instruction.

        Returns:
            Created SystemPrompt instance added to history.
        """

        message = SystemPrompt(content=prompt)
        self.append(message)
        return message

    def add_assistant_message(self, prompt: str) -> AssistantMessage:
        """
        Create and add an assistant's response to the conversation.

        Args:
            prompt: Text content of the assistant's reply.

        Returns:
            Created AssistantMessage instance with 'stop' finish reason.
        """

        message = AssistantMessage(content=prompt, finish_reason='stop')
        self.append(message)
        return message

    def add_tool_message(self, prompt: str, id: str) -> ToolCallResponse:
        """
        Create and add a tool response to the conversation history.

        Args:
            prompt: Output content from the executed tool.
            id: Unique identifier of the corresponding tool call.

        Returns:
            Created ToolCallResponse instance added to history.
        """

        message = ToolCallResponse(content=prompt, tool_call_id=id)
        self.append(message)
        return message

    @staticmethod
    def prepare_text_file(filepath: str, absprefix: str = '') -> str:
        """
        Wrap the contents of a file in XML-style tags for code context representation.

        Args:
            filepath: Path to the file to be wrapped.
            absprefix: Optional base directory path that will be stripped from the resulting filename display.

        Returns:
            A string containing the file content within <file>...</file> tags, with relative path metadata.
        """

        with open(filepath, 'r') as fin:
            content = fin.read()

        path = os.path.abspath(filepath)
        if absprefix != '':
            absprefix = os.path.abspath(absprefix)
            if path.startswith(absprefix):
                path = path[len(absprefix) :]

        nlines = content.count('\n')
        return f'''
<file name="{path}" from="1" to="{nlines}">
{content}
</file>
'''

    CODE_EXTS = ['.h', '.hpp', '.c', '.cpp', '.cu', '.txt', '.proto', '.py']

    @staticmethod
    def prepare_code_dir(
        root: str,
        exts: list[str] = CODE_EXTS,
        include_prefix: list[str] = [],
        exclude_prefix: list[str] = [],
        verbose: bool = False,
    ) -> str | list[str]:
        """
        Recursively wrap all relevant source code files in a directory structure for LLM context.

        Args:
            root: Base directory to start file scanning.
            exts: File extensions to include (default is CODE_EXTS, which contain common C++ and Python file extensions).
            include_prefix: List of subdirectory prefixes that should be included.
            exclude_prefix: List of subdirectory prefixes to explicitly exclude.
            verbose: If True, returns a list of individual file wrappers instead of concatenating into one string.

        Returns:
            Either concatenated XML-style project wrapper with all files, or list of individual wrapped files if verbose=True.
        """

        response = [f'<project root="{root}">']

        # Determine starting points
        if include_prefix:
            starts = [os.path.join(root, p) for p in include_prefix]
        else:
            starts = [root]

        seen = set()
        for start in starts:
            for dirpath, dirnames, filenames in os.walk(start):
                # Compute path relative to root, normalized to forward-slashes
                rel_dir = os.path.relpath(dirpath, root).replace('\\', '/')

                # Skip excluded prefixes
                if exclude_prefix and any(
                    rel_dir.startswith(p.rstrip('/')) for p in exclude_prefix
                ):
                    # Prevent descending further
                    dirnames[:] = []
                    filenames[:] = []
                    continue

                # Avoid duplicate walks if include_prefix overlap
                if dirpath in seen:
                    continue
                seen.add(dirpath)

                # Process files with matching extensions
                for file in filenames:
                    if any(file.endswith(ext) for ext in exts):
                        full_rel_path = os.path.join(rel_dir, file).replace('\\', '/')
                        if any(
                            full_rel_path.startswith(p.rstrip('/'))
                            for p in exclude_prefix
                        ):
                            continue

                        full_path = os.path.join(dirpath, file)
                        response.append(Chat.prepare_text_file(full_path, root))

        response.append('</project>')
        return '\n'.join(response) if not verbose else response[1:-1]

    def show_final_answer(
        self, hide_reasoning: bool = True, display_: bool = False
    ) -> Markdown:
        """
        Display the final assistant message as formatted Markdown.

        Args:
            hide_reasoning: If True, removes thought/reasoning blocks using remove_think_block.
            display_: If True, immediately displays the result in IPython environments.

        Returns:
            A Markdown object containing the processed final response.
        """

        last_message = self.history[-1].content
        if hide_reasoning:
            text = remove_think_block(last_message)
        md = Markdown(text)
        if display_:
            display(md)
        return md

    def __getitem__(self, key: int) -> AnyChatMessage:
        """Retrieve a specific message by index from the chat history."""
        return self.history[key]

    def get_text(self) -> str:
        """
        Generate plain text representation of entire conversation.

        Returns:
            Newline-separated string with role and content for each message.
        """

        return '\n'.join([f'{c.role}: {c.content}' for c in self.history])

    def display_thoughts(self, skip_reasoning: bool = False):
        """
        Display all messages in the conversation using appropriate formatting.

        Args:
            skip_reasoning: If True, skips displaying thought/reasoning blocks.
        """

        for message in self.history:
            display(display_message(message, skip_reasoning))

    def __iter__(self) -> Iterator[AnyChatMessage]:
        """Allow iteration over Chat to yield each message in history."""
        return iter(self.history)

    def __len__(self) -> int:
        """Return the number of messages in the chat history."""
        return len(self.history)
