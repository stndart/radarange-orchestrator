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

prompt = "Print a hello world message with python."

from simple_tool import code_tool

def dict_to_str(d: dict) -> str:
    s = ''
    for k in d.keys():
        s += f'\n{k}: {d[k]}'
    return s

conversation = [{"role": "user", "content": prompt}]
total_tokens = 0

response = llm.create_chat_completion(
    messages=conversation,
    tools=[code_tool],
    temperature=0.5,
    max_tokens=1000,
    stream=False,
)

print(response)