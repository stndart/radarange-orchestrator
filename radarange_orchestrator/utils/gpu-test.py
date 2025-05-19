import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"  # Prioritize GPU 1

import torch
print("Visible CUDA Devices:", torch.cuda.device_count())  # Should show 2
print("Current Device:", torch.cuda.current_device())  # Should match your priority
print(torch.cuda.set_current_device(1))