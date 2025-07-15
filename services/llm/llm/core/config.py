from importlib.util import find_spec

# Check if llama_cpp is installed
LLAMA_CPP_AVAILABLE = find_spec('llama_cpp') is not None

# Check if CUDA is available (only relevant if llama_cpp is installed)
CUDA_AVAILABLE = False
if LLAMA_CPP_AVAILABLE:
    try:
        from llama_cpp.llama import (
            LLAMA_BACKEND_CUDA,
            llama_available_devices,
            llama_backend_init,
        )

        llama_backend_init()
        CUDA_AVAILABLE = LLAMA_BACKEND_CUDA in llama_available_devices()
    except ImportError:
        # LLAMA_CPP_AVAILABLE = False
        pass

# Backend capabilities
BACKEND_CAPABILITIES = {
    'llama_cpp': {
        'available': LLAMA_CPP_AVAILABLE,
        'cuda': CUDA_AVAILABLE,
        'grammar': True,
    },
    'lmstudio': {
        'available': True,  # Always available
        'cuda': False,
        'grammar': False,
    },
}

LMSTUDIO_ADDRESS = '95.165.10.219'
LMSTUDIO_PORT = 1234

DEFAULT_LLM_MODEL = 'qwq-32b@q4_k_m'
