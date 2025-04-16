code_tool = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to run in the ipython interpreter."
                }
            },
            "required": [
                "code"
            ]
        }
    }
}

def dummy_tool(code: str):
    print(f"Dummy tool called with args: code = {code}")
    return len(code)

import subprocess
import shlex

def python(code: str) -> str:
    """
    Executes Python code in an IPython interpreter with a 60-second timeout.
    Returns combined stdout/stderr output or error message.
    """
    try:
        # Security note: This executes arbitrary code - use in sandboxed environment!
        process = subprocess.run(
            ['ipython', '--colors=NoColor', '-c', code],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = []
        if process.stdout: output.append(f"STDOUT:\n{process.stdout}")
        if process.stderr: output.append(f"STDERR:\n{process.stderr}")
        return "\n\n".join(output) or "No output"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 60 seconds"
    except FileNotFoundError:
        return "Error: IPython not installed"
    except Exception as e:
        return f"Error: {str(e)}"