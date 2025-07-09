import os
import time

import pymupdf
from langchain_core.tools import StructuredTool


def pdf_tool_handle(path: str) -> dict:
    print(f'Called pdf_read with path: {path}', flush=True)
    ts = time.time()

    doc = pymupdf.open(path)
    text = ''
    for page in doc:  # iterate the document pages
        for block in page.get_text('blocks'):
            text += block[4]
    images = []
    image_n = 0
    images_path = os.path.join(os.path.dirname(path), 'images')
    os.makedirs(images_path, exist_ok=True)
    for page in doc:
        for image in page.get_images():
            fn = os.path.join(images_path, f'im_{image_n}.png')
            image_n += 1
            pymupdf.Pixmap(doc, image[0]).save(fn)
            images.append(fn)

    result = {'content': text, 'images': images}

    print(f'Taken {time.time() - ts:.1f} seconds to complete.')
    return result


TOOLNAME = 'pdf_read'

pdf_tool = StructuredTool.from_function(
    name=TOOLNAME,
    func=pdf_tool_handle,
    description='Opens and converts pdf file to markdown text. \
        Accepts single argument: path to file. \
        Returns the list of extracted images filenames and the text content of the file. \
        Note that formula-related text can be wrecked',
)
