from IPython.display import Markdown, display

from .types.history import AnyChatMessage, UserMessage, SystemPrompt, AssistantMessage, ToolCallResponse
from .types.tools import Tool, ToolHandler
from .utils import display_message, is_list_of, make_tool_from_fun
from .utils.extract_tool_calls import remove_think_block
class Chat:
    history: list[AnyChatMessage]
    tools: list[Tool]

    def __init__(
        self,
        prompt: str | AnyChatMessage = '',
        tools: list[ToolHandler] | list[Tool] = [],
    ):
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

    def copy(self) -> 'Chat':
        new_chat = Chat()
        new_chat.tools = self.tools  # no need to copy since tools are const
        for i, message in enumerate(self.history):
            new_chat.append(message.model_copy(deep=True))

        return new_chat

    def append(self, message: AnyChatMessage):
        self.history.append(message)

    def add_user_message(self, prompt: str) -> UserMessage:
        message = UserMessage(content=prompt)
        self.append(message)
        return message

    def add_system_message(self, prompt: str) -> SystemPrompt:
        message = SystemPrompt(content=prompt)
        self.append(message)
        return message

    def add_assistant_message(self, prompt: str) -> AssistantMessage:
        message = AssistantMessage(content=prompt, finish_reason='stop')
        self.append(message)
        return message

    def add_tool_message(self, prompt: str, id: str) -> ToolCallResponse:
        message = ToolCallResponse(content=prompt, tool_call_id=id)
        self.append(message)
        return message

    def show_final_answer(self, hide_reasoning: bool = True) -> Markdown:
        last_message = self.history[-1].content
        if hide_reasoning:
            text = remove_think_block(last_message)
        md = Markdown(text)
        display(md)
        return md

    def __getitem__(self, key: int) -> AnyChatMessage:
        return self.history[key]

    def get_text(self) -> str:
        return '\n'.join([f'{c.role}: {c.content}' for c in self.history])

    def display_thoughts(self, skip_reasoning: bool = False):
        for message in self.history:
            display(display_message(message, skip_reasoning))

    def __iter__(self):
        """Allow iteration over Chat to yield each message in history."""
        return iter(self.history)

    def __len__(self):
        """Return the number of messages in the chat history."""
        return len(self.history)
