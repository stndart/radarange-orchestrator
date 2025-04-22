from typing import TypeVar
import re
from IPython.display import display, Markdown, HTML

from .llm_types import ChatMessage, MessageType, Response, ToolCallResponse
from llama_cpp.llama_types import ChatCompletionRequestMessage
from .tools.tool_annotation import Tool, ToolHandler
from .utils import is_list_of, make_tool_from_fun
from .llama_cpp_bindings import to_llama_message

class Chat:
    def __init__(self, prompt: str | ChatMessage = "", tools: list[ToolHandler] | list[Tool] = []):
        self.history: list[MessageType] = []
        
        if isinstance(prompt, str):
            if prompt != "":
                self.add_user_message(prompt)
        else:
            self.append(prompt)
        
        self.tools: list[Tool] = []
        if len(tools) > 0:
            if is_list_of(tools, Tool):
                self.tools = tools
            else:
                self.tools = [make_tool_from_fun(fun) for fun in tools] # type: ignore
    
    def copy(self) -> "Chat":
        new_chat = Chat()
        new_chat.tools = self.tools  # no need to copy since tools are const
        for message in self.history:
            new_chat.append(message.model_copy(deep=True))
        
        return new_chat

    T = TypeVar('T', bound=MessageType)
    def append(self, message: T) -> T:
        self.history.append(message)
        return message

    def add_user_message(self, prompt: str) -> ChatMessage:
        return self.append(ChatMessage(role="user", content=prompt))

    def add_system_message(self, prompt: str) -> ChatMessage:
        return self.append(ChatMessage(role="system", content=prompt))

    def add_assistant_message(self, prompt: str) -> Response:
        return self.append(Response(role="assistant", content=prompt, finish_reason="stop"))

    def add_tool_message(self, prompt: str, id: str) -> ToolCallResponse:
        return self.append(ToolCallResponse(role="tool", content=prompt, tool_call_id=id))

    def llama_messages(self) -> list[ChatCompletionRequestMessage]:
        return [to_llama_message(m) for m in self.history]

    def show_final_answer(self, hide_reasoning: bool = True):
        last_message = self.history[-1].content
        text = re.sub(r'<think>.*?</think>', '', last_message, flags=re.DOTALL)
        display(Markdown(text))