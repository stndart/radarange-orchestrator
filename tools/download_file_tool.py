from .tool_annotation import ToolResult, ToolType, Tool

download_tool_def = ToolType(**{
    "type": "function",
    "function": {
        "name": "download_file",
        "description": "Retrieves and saves the contents of a file from a specified URL. Supports common file formats like PDF, DOCX, images, and binaries.",
        "parameters": {
            "type": "object",
            "properties": {
                "href": {
                    "type": "string",
                    "description": "Full URL (including protocol) of the file to download. Must point directly to a downloadable resource."
                },
                "filename": {
                    "type": "string",
                    "description": "Target download file name (including extension). The file is going to be placed at the ./downloads/`filename` path."
                }
            },
            "required": ["href", "filename"]
        }
    }
})

import requests
import os

def handle_download_tool(href: str, filename: str) -> ToolResult:
    print(f"Called download_file with href: {href} and filename: {filename}", flush=True)
    
    result = ToolResult(**{"status": "success", "stdout": "", "stderr": "", "returncode": 0})
    
    filename = os.path.join("downloads", filename)
    
    # Validate URL protocol
    if not href.startswith(('http://', 'https://')):
        result.status = "error"
        result.stdout = f"Invalid URL protocol: {href}"
        result.returncode = 1
        return result

    try:
        # Download file with streaming
        with requests.get(href, stream=True, timeout=10) as response:
            response.raise_for_status()
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Write file in chunks
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Add success message
            file_size = os.path.getsize(filename)
            result.stdout = f"Downloaded {file_size} bytes to {filename}"

    except requests.exceptions.RequestException as e:
        result.status = "error"
        result.stdout = f"Download failed: {str(e)}"
        result.returncode = 1
    except IOError as e:
        result.status = "error"
        result.stdout = f"File write error: {str(e)}"
        result.returncode = 1
    except Exception as e:
        result.status = "error"
        result.stdout = f"Unexpected error: {str(e)}"
        result.returncode = 1

    return result

download_tool = Tool(**{
    "definition": download_tool_def,
    "handler": handle_download_tool
})