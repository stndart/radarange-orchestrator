# ~/llm/infer.py
import os, time
from typing import Iterator
from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_ROW, llama_types

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Both GPUs

MODEL_PATH = "QwQ-32B-Q4_K_M.gguf"
CONTEXT_SIZE = int(100e3)  # for both gpus
# CONTEXT_SIZE = int(10e3)   # for one gpu
SPLIT_MODE = LLAMA_SPLIT_MODE_LAYER

tensor_split = [0.37176396722521365,0.3910968437984273] # from lm_studio logs

start_time = time.time()  # Record start time before the API call
llm = Llama(
    model_path=os.path.expanduser(MODEL_PATH),
    n_gpu_layers=-1, # all
    split_mode=SPLIT_MODE, # the most optimal for speed
    main_gpu = 1, # ignored for layer split mode
    tensor_split=tensor_split, # balancing layers between gpus
    n_ctx = CONTEXT_SIZE, # 100k context
    n_batch = 4096, # todo: make tests
    n_threads = 6, # cpu threads
    flash_attn = True, # Flash attention # don't know if works for QwQ:32b
    numa = 3, # Optimize NUMA allocation
    verbose = False,
    no_perf = False
)

prompt = "Write a single-file app that prints current time and tell me, what time is it."

from ..tools.code_writing import code_tool, handle_code_tool

def process_response(response):
    message = response['choices'][0]['message']
    
    # Check for tool call
    if 'tool_calls' in message:
        for tool_call in message['tool_calls']:
            if tool_call['function']['name'] == 'write_python_code':
                args = json.loads(tool_call['function']['arguments'])
                if not validate_code(args['content']):
                    print(f"Code rejected: dangerous code detected")
                    return {"status": "rejected", "reason": "Dangerous code detected"}
                
                result = handle_code_tool(
                    args['filename'],
                    args['content'],
                    args.get('execute', False),
                    args.get('timeout', 30)
                )
                # print(f"Code execution result: {result}")
                return result

# Enhanced safety checks (basic example)
def validate_code(content):
    forbidden_patterns = [
        r"os\.system",
        r"subprocess",
        r"open\(.*, ['\"]w['\"]\)",
        r"import\s+os",
        r"import\s+subprocess"
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, content):
            return False
    return True

def dict_to_str(d: dict) -> str:
    s = ''
    for k in d.keys():
        s += f'\n{k}: {d[k]}'
    return s

conversation = [{"role": "user", "content": prompt}]
total_tokens = 0

while True:
    response = llm.create_chat_completion(
        messages=conversation,
        tools=[code_tool],
        temperature=0.5,
        max_tokens=1000,
        stream=False,
    )

    total_tokens += response['usage']['completion_tokens']
    result = process_response(response)
    
    # Add both AI message and tool result to conversation
    message = response['choices'][0]['message']
    conversation.append(message)
    
    if result is not None:
        conversation.append({
            "role": "tool",
            "content": f"Execution result:{dict_to_str(result)}",
            "name": "write_python_code"
        })
    else:
        break

end_time = time.time()  # Record end time after the API response

# Calculate metrics
elapsed_seconds = end_time - start_time
tokens_per_second = total_tokens / elapsed_seconds if elapsed_seconds != 0 else 0.0

print(f"Total tokens: {total_tokens}")
print(f"Time taken (seconds): {elapsed_seconds:.2f}")
print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")

print("\n\nFinal response:")
result = ''
for tmessage in conversation:
    result += f"{tmessage['role']}: {tmessage['content']}\n"
print(result)