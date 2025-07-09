from typing import Annotated, Callable, Optional, TypeVar, Union

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.utils import _get_type
from pydantic import Discriminator, Field, Tag

AnyCompleteMessage = Annotated[
    Union[
        Annotated[AIMessage, Tag(tag='ai')],
        Annotated[HumanMessage, Tag(tag='human')],
        Annotated[SystemMessage, Tag(tag='system')],
        Annotated[ToolMessage, Tag(tag='tool')],
    ],
    Field(discriminator=Discriminator(_get_type)),
]

AnyMessageChunk = Annotated[
    Union[
        Annotated[AIMessageChunk, Tag(tag='AIMessageChunk')],
        Annotated[HumanMessageChunk, Tag(tag='HumanMessageChunk')],
        Annotated[SystemMessageChunk, Tag(tag='SystemMessageChunk')],
        Annotated[ToolMessageChunk, Tag(tag='ToolMessageChunk')],
    ],
    Field(discriminator=Discriminator(_get_type)),
]

AnyMessage = AnyMessageChunk | AnyCompleteMessage


T = TypeVar('T', bound=AnyCompleteMessage)
MessageHandler = Callable[[T], Optional[T]]


def EmptyMessageHandler(response: AIMessage):
    pass
