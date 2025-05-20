from typing import Any, Iterator, Optional

from ..types.tools import Tool
from ..types.history import AssistantMessage, AssistantMessageFragment
from ..chat import Chat


class GenericModel:
    def create_chat_completion(
        self,
        chat: Chat,
        tools: list[Tool],
        response_format: Optional[dict[str, str]] = None,
        grammar: Optional[Any] = None,  # TODO
        temperature: float = 0.7,
        max_tokens: int = 5000,
        stream: bool = False,
    ) -> AssistantMessage | Iterator[AssistantMessageFragment]:
        pass
