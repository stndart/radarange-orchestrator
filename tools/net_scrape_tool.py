import time

import requests
from html2text import HTML2Text
from readability import Document

from .tool_annotation import (
    FunctionDescription,
    ParameterProperty,
    Parameters,
    Tool,
    ToolResult,
    ToolType,
)

scrape_tool_def = ToolType(
    type="function",
    function=FunctionDescription(
        name="scrape_web_page",
        description="Fetches and returns the full text content of a web page from a specified URL. Optionally truncates the content to a specified character length.",
        parameters=Parameters(
            type="object",
            properties={
                "href": ParameterProperty(
                    type="string",
                    description="The full URL (including protocol) of the web page to scrape content from.",
                ),
                "truncate_content": ParameterProperty(
                    type="number",
                    description="Optional maximum character length for the returned content. Content exceeding this limit will be truncated. 10000 characters by default.",
                ),
            },
            required=["href"],
        ),
    ),
)


def get_clean_page_content(url: str, n_truncate: int):
    try:
        # 1. Fetch page
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # 2. Extract main content
        doc = Document(response.text)
        cleaned_html = doc.summary()

        # 3. Convert to clean Markdown
        h = HTML2Text()
        h.ignore_links = False
        markdown_content = h.handle(cleaned_html)

        # 4. Truncate for LLM context
        return markdown_content[:n_truncate]

    except Exception as e:
        return f"Failed to retrieve content: {str(e)}"


def handle_scrape_tool(
    href: str, truncate_content: int = 10000, source: str = "inference"
) -> ToolResult:
    if source == "inference":
        print(f"Called scrape_web_page with href: {href}", flush=True)
    ts = time.time()
    result = ToolResult(
        **{
            "status": "success",
            "stdout": get_clean_page_content(href, truncate_content),
            "stderr": "",
            "returncode": 0,
        }
    )
    if source == "inference":
        print(f"Taken {time.time() - ts:.1f} seconds to complete.")
        if result.status == "error":
            if len(result.stderr) > 150:
                print(f"Tool evaluation led to error: {result.stderr[:100]} ... {result.stderr[-50:]}")
            else:
                print(f"Tool evaluation led to error: {result.stderr}")
    return result


scrape_tool = Tool(**{"definition": scrape_tool_def, "handler": handle_scrape_tool})
