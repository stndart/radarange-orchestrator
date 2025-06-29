import lmstudio as lms
from lmstudio._sdk_models import LlmLoadModelConfig, GpuSetting

def multiply(a: float, b: float) -> float:
    """Given two numbers a and b. Returns the product of them."""
    return a * b


with lms.Client("95.165.10.219:1234") as client:
    model = client.llm.model('qwen3-32b')
    model.act(
        'What is the result of 12345 multiplied by 54321?',
        [multiply],
    )
