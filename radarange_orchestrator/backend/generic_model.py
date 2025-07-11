from typing import Iterator, Optional

from ..formatting import ResponseFormat
from ..chat import Chat, AIMessage, AIMessageChunk
from ..tools import Tool


class GenericModel:
    def create_chat_completion(
        self,
        chat: Chat,
        tools: list[Tool],
        response_format: Optional[ResponseFormat] = None,
        temperature: float = 0.7,
        max_tokens: int = 5000,
        stream: bool = False,
    ) -> AIMessage | Iterator[AIMessageChunk]:
        pass

    def count_tokens(self, prompt: str | Chat) -> int:
        pass

    def close(self) -> None:
        pass

    def assure_loaded(self) -> None:
        pass