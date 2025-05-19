import os
from llama_cpp import Llama

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import torch
# print("Visible CUDA Devices:", torch.cuda.device_count())  # Should show 2
# print("Current Device:", torch.cuda.current_device())  # Should match your priority

mp = "./models/QwQ-32B-Q4_K_M.gguf"

model = Llama(
    model_path=mp,
    n_gpu_layers=-1,
)
