# ~/llm/infer.py
import os, time
from typing import Iterator
from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_ROW, llama_types

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Both GPUs

import torch
print("Visible CUDA Devices:", torch.cuda.device_count())  # Should show 2
print("Current Device:", torch.cuda.current_device())  # Should match your priority

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

from ..tools.code_writing import code_tool

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=500,
    tools=[code_tool],
    stream=False,
    function_call = lambda token, _: tester.callback(token, _)
)

if (isinstance(response, Iterator)):
    full_response = ''
    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            token_text = delta["content"]
            full_response += token_text
            # Print tokens as they arrive (optional)
            # print(token_text, end="", flush=True)

    print("\n\nFinal response:")
    print(full_response)
else:
    end_time = time.time()  # Record end time after the API response
    
    # Calculate metrics
    total_tokens = response['usage']['completion_tokens']
    elapsed_seconds = end_time - start_time
    tokens_per_second = total_tokens / elapsed_seconds if elapsed_seconds != 0 else 0.0

    print("Usage:", response['usage'])
    print(f"Total tokens: {total_tokens}")
    print(f"Time taken (seconds): {elapsed_seconds:.2f}")
    print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")

    print("\n\nFinal response:")
    print(response['choices'][0]['message'])