from functools import wraps
from typing import Iterator, Sequence

from IPython.display import Markdown, display
from langchain_core.chat_history import InMemoryChatMessageHistory

from ..tools import Tool
from ..utils import display_message
from ..utils.extract_tool_calls import remove_think_block
from .messages import AnyCompleteMessage


class Chat(InMemoryChatMessageHistory):
    """
    Iterable container for AnyCompleteMessage. Stores messages and tools assigned to this context.
    
    Has additional output methods:
    - show_final_answer - exports last message to markdown
    - get_text - exports full text to string
    - display_thoughts - renders HTML representation of chat (in jupyter notebook)
    """

    tools: list[Tool] = []

    @wraps(InMemoryChatMessageHistory.add_message)
    def add_message(self, message: AnyCompleteMessage) -> None:
        super().add_message(message)

    @wraps(InMemoryChatMessageHistory.add_messages)
    def add_messages(self, messages: Sequence[AnyCompleteMessage]) -> None:
        super().add_messages(messages)

    @wraps(InMemoryChatMessageHistory.aadd_messages)
    async def aadd_messages(self, messages: Sequence[AnyCompleteMessage]) -> None:
        super().aadd_messages(messages)

    def __iter__(self) -> Iterator[AnyCompleteMessage]:
        return iter(self.messages)

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

        last_message = self.messages[-1].content
        if hide_reasoning:
            text = remove_think_block(last_message)
        md = Markdown(text)
        if display_:
            display(md)
        return md

    def __getitem__(self, key: int) -> AnyCompleteMessage:
        """Retrieve a specific message by index from the chat history."""
        return self.messages[key]

    def get_text(self) -> str:
        """
        Generate plain text representation of entire conversation.

        Returns:
            Newline-separated string with role and content for each message.
        """

        return '\n'.join([f'{c.type}: {c.content}' for c in self.messages])

    def display_thoughts(self, skip_reasoning: bool = False):
        """
        Display all messages in the conversation using appropriate formatting.

        Args:
            skip_reasoning: If True, skips displaying thought/reasoning blocks.
        """
        raise NotImplementedError('Fix annotations before using this')

        for message in self.messages:
            display(display_message(message, skip_reasoning))
