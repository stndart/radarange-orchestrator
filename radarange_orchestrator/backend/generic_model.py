from typing import Iterator, Optional

from radarange_orchestrator.formatting import ResponseFormat

from ..types.tools import Tool
from ..types.history import AssistantMessage, AssistantMessageFragment
from ..chat import Chat


class GenericModel:
    def create_chat_completion(
        self,
        chat: Chat,
        tools: list[Tool],
        response_format: Optional[ResponseFormat] = None,
        temperature: float = 0.7,
        max_tokens: int = 5000,
        stream: bool = False,
    ) -> AssistantMessage | Iterator[AssistantMessageFragment]:
        pass

    def count_tokens(self, prompt: str | Chat) -> int:
        pass