from typing import Optional
from .llm_types import Response, MessageType, ChatMessage, ToolCallResponse, ToolCall
from .tools.tool_annotation import Tool, ToolCallFunction
from llama_cpp.llama_types import CreateChatCompletionResponse, ChatCompletionMessageToolCall, ChatCompletionTool, ChatCompletionRequestMessage, ChatCompletionMessageToolCalls 
from .utils.extract_tool_calls import extract_tool_calls

def fill_tool_calls(tool_calls: list[ToolCall], text: str) -> list[ToolCall]:
    return tool_calls + extract_tool_calls(text)

def from_chat_completion(model_response: CreateChatCompletionResponse) -> Response:
    def none_to_str(text: Optional[str]) -> str:
        return text if text else ''

    tool_calls = model_response["choices"][0]["message"].get("tool_calls", [])
    role = model_response["choices"][0]["message"]["role"]
    assert role == 'assistant'
    return Response(
        content = none_to_str(model_response["choices"][0]["message"]["content"]),
        tool_calls = fill_tool_calls(
            [from_llama_tool_call(call) for call in tool_calls], 
            none_to_str(model_response["choices"][0]["message"]["content"])
        ),
        finish_reason = model_response["choices"][0]["finish_reason"],
        role = role
    )

def to_llama_tool_call(tool_call: ToolCall) -> ChatCompletionMessageToolCall:
    return {
        "type": "function",
        "id": tool_call.id,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }
    }

def from_llama_tool_call(tool_call: ChatCompletionMessageToolCall) -> ToolCall:
    return ToolCall(
        id = tool_call["id"],
        type = "function",
        function = ToolCallFunction(
            name = tool_call["function"]["name"],
            arguments = tool_call["function"]["arguments"]
        )
    )

def to_llama_tools(tools: list[Tool]) -> list[ChatCompletionTool]:
    return [{
        "type": "function",
        "function": {
            "name": tool.definition.function.name,
            "description": tool.definition.function.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": prop.type,
                        "description": prop.description
                    } for name, prop in tool.definition.function.parameters.properties.items()
                },
                "required": tool.definition.function.parameters.required
            }
        }
    } for tool in tools ]

def to_llama_message(message: MessageType) -> ChatCompletionRequestMessage:
    if isinstance(message, ChatMessage):
        if message.role == "system":
            return {
                "role": "system",
                "content": message.content
            }
        else: # message.role == "user":
            return {
                "role": "user",
                "content": message.content
            }
    elif isinstance(message, ToolCallResponse):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id
        }
    else:  # Response
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [to_llama_tool_call(call) for call in message.tool_calls]
        }
