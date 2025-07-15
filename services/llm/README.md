cd orchestrator

# 1. Create a fresh venv with Python 3.12
uv venv .venv --python 3.12
source .venv/bin/activate

# 2. Install the LLM service in editable mode
cd services/llm
uv pip install -e .[local,dev]

# 3. Register Jupyter kernel for use in notebooks
uv run python -m ipykernel install --user --name radarange-llm --display-name "Radarange LLM"
