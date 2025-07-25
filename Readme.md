# radarange-orchestrator

This package provides orchestration tools for [presumably a larger project, specify the domain if known e.g., radar data processing, distributed computations, etc.].

**prerequisities**:
*   **uv:** A fast Python package installer and resolver. (install with `pip install uv`)

## Installation [default]

In this mode, you reach the LM Studio server over internet

Installation steps:

1.  Create a virtual environment:
    ```bash
    uv venv .venv
    ```

2.  Install the package in editable mode:
    ```bash
    uv pip install -e .
    ```

3.  Configure LM Studio server address and port in radarange_orchestrator/config.py

## Installation [local]

Before installing, ensure you have the following prerequisites:

*   **Visual Studio Build Tools:** Required for compiling native extensions.
*   **CUDA:** Needed for GPU acceleration (if applicable).

Installation steps:

1.  Create a virtual environment:
    ```bash
    uv venv .venv
    ```

2.  Install the package in editable mode:
    ```bash
    uv pip install -e .[local]
    ```

2.  Open any jupyter notebook, for example, notebooks/base.ipynb, and when prompted, select newly created .venv as python interpreter path

## Examples

Example usage can be found in the `examples/` directory.

Example can be run via
```bash
uv run python examples/example.py
```