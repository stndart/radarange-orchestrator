# ~/llm/download_model.py
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="lmstudio-community/QwQ-32B-GGUF",
    filename="QwQ-32B-Q4_K_M.gguf",
    local_dir="~/llm",
    local_dir_use_symlinks=False,
    force_download=True
)
