from .tool_annotation import *

code_tool = ToolType(**{
    "type": "function",
    "function": {
        "name": "write_python_code",
        "description": "Write Python code to a file and optionally execute it",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename including .py extension"
                },
                "content": {
                    "type": "string",
                    "description": "Full Python code content with proper indentation"
                },
                "execute": {
                    "type": "boolean",
                    "description": "Whether to immediately execute the code"
                },
                "timeout": {
                    "type": "number",
                    "description": "Program execution timeout. If expired, program would be terminated immediately. -1 means no timeout."
                }
            },
            "required": ["filename", "content"]
        }
    }
})

import subprocess
import os

CODE_DIR = "generated_code"

def handle_code_tool(filename, content, execute=False, timeout=30) -> ToolResult:
    print(f"Called write_python_code with filename: {filename}", flush=True)
    
    # Create code directory if needed
    os.makedirs(CODE_DIR, exist_ok=True)
    print('code is placed at', CODE_DIR)
    
    if timeout == -1:
        timeout = None

    # Write code to file
    filepath = os.path.join(CODE_DIR, filename)
    with open(filepath, "w") as f:
        f.write(content)
    
    result = {"status": "written", "filepath": filepath}
    
    if execute:
        try:
            # Execute code safely with timeout
            process = subprocess.run(
                ["python", filepath],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            result.update({
                "status": "executed",
                "stdout": process.stdout,
                "stderr": process.stderr,
                "returncode": process.returncode
            })
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e)
            })
    
    return result