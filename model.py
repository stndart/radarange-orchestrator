import os, time, re, json, uuid
from typing import Optional, Union, Callable
from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_ROW, llama_types as lt

from .tools.tool_annotation import *

class MyModel:
    def __init__(self, model_path: str = "QwQ-32B-Q4_K_M.gguf",
                 gpus: list[int] = [0, 1], ctx_size: Optional[int] = None, split_mode = LLAMA_SPLIT_MODE_LAYER,
                 tools: list[Tool] = []):
        self.MODEL_PATH = model_path
        self.GPUS = gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        self.SPLIT_MODE = split_mode
        
        self.CONTEXT_SIZE = ctx_size
        # fit all the context and weights on gpu to perform 30 t/s
        if not ctx_size:
            if len(gpus) == 1:
                self.CONTEXT_SIZE = int(10e3)
            elif len(gpus) == 2:
                self.CONTEXT_SIZE = int(80e3)
        
        tensor_split = [1.0]
        if len(gpus) == 2:
            tensor_split = [0.37176396722521365,0.3910968437984273] # from lm_studio logs
        
        self.tools = tools
        
        model_path = self.find_model(model_path)

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # all
            split_mode=self.SPLIT_MODE, # the most optimal for speed
            main_gpu = 1, # ignored for layer split mode
            tensor_split=tensor_split, # balancing layers between gpus
            n_ctx = self.CONTEXT_SIZE,
            n_batch = 4096, # todo: make tests
            n_threads = 6, # cpu threads
            flash_attn = True, # Flash attention # don't know if works for QwQ:32b
            numa = 3, # Optimize NUMA allocation
            verbose = False
        )
    
    def find_model(self, path: str):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to your model file
        MODEL_PATH = os.path.join(BASE_DIR, 'models', path)
        return MODEL_PATH
        
    def create_completion(self, prompt: str, max_tokens: int = 500, temperature: float = 0.5, stream: bool = False,
                                 stop: Optional[Union[str, list[str]]] = []):
        """
        Simple wrapper for Llama.create_completion. Doesn't support tools nor roles.
        """
        return self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=stream
        )
    
    def create_simple_chat_completion(self, prompt: str, max_tokens: int = 500, temperature: float = 0.5, stream: bool = False,
                                 stop: Optional[Union[str, list[str]]] = [], previous_messages: list[LLMMessage] = []) -> tuple[list[LLMMessage], str]:
        """
        Default behaviour. Creates next chat message for the prompt with user role. Pass previous_messages param to continue the chat.
        Returns:
            - messages: list[LLMMessage]
            - conversation: str (does not include previous_messages)
        """
        messages: list[LLMMessage] = previous_messages + [
            {"role": "user", "content": prompt}
        ]
        conversation = f'user: {prompt}'
        while True:
            response = self.llm.create_chat_completion(
                messages=messages,
                tools=get_tool_defs(self.tools),
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=stream
            )
            if stream:
                break
            else:
                messages.append(response)
                conversation += f'\n\n assistant: {response["choices"][0]["message"]["content"]}'
                if response["choices"][0]["finish_reason"] == "stop": # like an EOF
                    break
        return messages, conversation
    
    def add_tool(self, tool: Tool):
        """
        Adds tool to model accessible tools.
        Tools are used with:
            - create_simple_chat_completion: though no parsing is performed
            - create_chat_completion_with_tool: parses and optionally executes tool calls
        """
        self.tools.append(tool)
    
    tool_call_counter: int = 0
    def extract_tool_calls(self, text: str, skip_reasoning: bool = True) -> lt.ChatCompletionMessageToolCalls:
        """
        Extracts last <tool_call></tool_call> block from text and optionally executes it
        """
        def invalid_tool_call(tool_call_counter: int, name: str = 'invalid') -> lt.ChatCompletionMessageToolCall:
            return {
                'id': f'invalid_{tool_call_counter}',
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': dict()
                    }
                }
        
        # If we need to skip <think> blocks, remove them from the text
        if skip_reasoning:
            # This will remove everything from <think> to </think>, including newlines.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        
        tag_begin, tag_end = "<tool_call>", "</tool_call>"
        results: lt.ChatCompletionMessageToolCalls = []
        for tool_block in re.findall(rf"{re.escape(tag_begin)}(.*?){re.escape(tag_end)}", text, re.DOTALL):
            MyModel.tool_call_counter += 1
            try:
                # Parse the JSON content of the tool call
                tool_data = json.loads(tool_block)
            except json.JSONDecodeError:
                results.append(invalid_tool_call(MyModel.tool_call_counter))
            else:
                tool_name = tool_data.get('name')
                arguments = tool_data.get('arguments', {})
                if not tool_name:
                    results.append(invalid_tool_call(MyModel.tool_call_counter))
                    continue
                
                tool_handler: ToolHandler = get_tool_handler(self.tools, tool_name)
                if not tool_handler:
                    results.append(invalid_tool_call(MyModel.tool_call_counter, tool_name))
                    continue
                
                results.append({
                    'id': f'{tool_name}_{MyModel.tool_call_counter}',
                    'type': 'function',
                    'function': {
                        'name': tool_name,
                        'arguments': arguments
                    }
                })
        
        return results
    
    def process_tool_calls(self, text: str, execute: bool = True) -> tuple[lt.ChatCompletionMessageToolCalls, Optional[list[ToolResult]]]:
        """
        Extracts all <tool_call></tool_call> block from text and optionally executes them
        """
        
        tool_calls: lt.ChatCompletionMessageToolCalls = self.extract_tool_calls(text)
        print(f"Extracted {len(tool_calls)} tool_calls, execution = {execute}")
        if not execute:
            return tool_calls, None
        
        self.state = tool_calls
        results: list[ToolResult] = []
        for tool_call in tool_calls:
            # print(tool_call)
            if 'invalid' in tool_call["id"]:
                # print("invalid tool call")
                if 'invalid' in tool_call["function"]["name"]:
                    results.append({
                        'status': 'Parsing error',
                        'stdout': '',
                        'stderr': 'Tool call missing \'name\' field',
                        'returncode': -1
                    })
                else:
                    name = tool_call["function"]["name"]
                    results.append({
                        'status': 'Parsing error',
                        'stdout': '',
                        'stderr': f"Error: Tool '{name}' not found.",
                        'returncode': -1
                    })
            else:
                name = tool_call["function"]["name"]
                tool_handler = get_tool_handler(self.tools, name)
                # print(f"executing {name} tool")
                
                # Execute the tool handler with the provided arguments
                stdout = ''
                stderr = ''
                returncode = -1
                try:
                    tool_result = tool_handler(**tool_call["function"]["arguments"])
                    stdout = json.dumps(tool_result, indent=2)
                    returncode = 0
                except Exception as e:
                    stderr = f"Error executing tool '{name}': {str(e)}"
                    returncode = -1
                finally:
                    results.append({
                        'status': 'success' if stderr == '' else 'error',
                        'stdout': stdout,
                        'stderr': stderr,
                        'returncode': returncode
                    })
        
        # print(results)
        return tool_calls, results
    
    def create_chat_completion_with_tool(self, prompt: str, max_tokens: int = 500, temperature: float = 0.5, stream: bool = False,
                                 stop: Optional[Union[str, list[str]]] = [], tool_role: str = 'tool',
                                 max_iterations: int = 10, previous_messages: list[dict[str, str]] = []):
        # is not supported for now
        assert not stream
        
        messages = previous_messages + [
            {"role": "user", "content": prompt}
        ]
        conversation = f'user: {prompt}'

        i = max_iterations
        while i > 0:
            i -= 1
            response = self.llm.create_chat_completion(
                messages=messages,
                tools=get_tool_defs(self.tools),
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=stream
            )
            messages.append(response)
            finish_reason = response["choices"][0]["finish_reason"]
            text = response["choices"][0]["message"]["content"]
            conversation += f'\n\nassistant:\n{text}'
            
            tool_calls, tool_results = self.process_tool_calls(text)
            if len(tool_calls) > 0:
                finish_reason = 'tool_call'
            
            for call, result in zip(tool_calls, tool_results):
                message = {
                    "role": tool_role,
                    "content": json.dumps(result),
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"]
                }
                
                conversation += f'\n\ntool:\n<tool_call_result>{message}</tool_call_result>'
                messages.append(message)
            
            if finish_reason == "stop": # like an EOF
                break
        return messages, conversation