import os
import time

import requests
from langchain_core.tools import StructuredTool


def download(href: str, filename: str) -> str:
    print(
        f'Called download_file with href: {href} and filename: {filename}', flush=True
    )
    ts = time.time()
    filename = os.path.join('downloads', filename)

    # Validate URL protocol
    if not href.startswith(('http://', 'https://')):
        raise RuntimeError(f"Invalid URL protocol: {href}")

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
        stdout = f'Downloaded {file_size} bytes to {filename}'

    print(f'Taken {time.time() - ts:.1f} seconds to complete.')
    return stdout


TOOLNAME = 'download_file'

download_tool = StructuredTool.from_function(
    name=TOOLNAME,
    func=download,
    description='Retrieves and saves the contents of a file from a specified URL. Supports common file formats like PDF, DOCX, images, and binaries. \
        Arguments:\
            - href: string -Full URL (including protocol) of the file to download. Must point directly to a downloadable resource.\
            - filename: string - Target download file name (including extension). The file is going to be placed at the ./downloads/`filename` path.',
)