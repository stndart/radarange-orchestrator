import os
import time

import pymupdf

from .tool_annotation import (
    FunctionDescription,
    ParameterProperty,
    Parameters,
    Tool,
    ToolResult,
    ToolType,
)

pdf_tool_def = ToolType(
    type="function",
    function=FunctionDescription(
        name="pdf_read",
        description="Opens and converts pdf file to text. Returns the list of extracted images filenames and the text content of the file. Note that formula-related text can be wrecked",
        parameters=Parameters(
            type="object",
            properties={
                "path": ParameterProperty(
                    type="string",
                    description="Path to pdf file",
                ),
            },
            required=[],
        ),
    ),
)


def pdf_tool_handle(path: str) -> ToolResult:
    print(f"Called pdf_read with path: {path}", flush=True)
    ts = time.time()
    try:
        doc = pymupdf.open(path)
        text = ""
        for page in doc:  # iterate the document pages
            for block in page.get_text("blocks"):
                text += block[4]
        images = []
        image_n = 0
        images_path = os.path.join(os.path.dirname(path), "images")
        os.makedirs(images_path, exist_ok=True)
        for page in doc:
            for image in page.get_images():
                fn = os.path.join(images_path, f"im_{image_n}.png")
                image_n += 1
                pymupdf.Pixmap(doc, image[0]).save(fn)
                images.append(fn)

        result = ToolResult(
            status="success",
            stdout=f"images: {images};\ncontent: {text}",
            stderr="",
            returncode=0,
        )
    except Exception as e:
        result = ToolResult(
            status="error",
            stdout="",
            stderr=f"Pdf parse error: {e}\nTraceback: {e.__traceback__}",
            returncode=-1,
        )

    print(f"Taken {time.time() - ts:.1f} seconds to complete.")
    return result


pdf_tool = Tool(definition=pdf_tool_def, handler=pdf_tool_handle)
