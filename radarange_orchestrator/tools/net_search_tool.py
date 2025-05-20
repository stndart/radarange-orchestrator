import json
import time

from duckduckgo_search import DDGS

from .net_scrape_tool import handle_scrape_tool
from ..types.tools import (
    FunctionDescription,
    ParameterProperty,
    Parameters,
    Tool,
    ToolResult,
    ToolDef,
)

net_tool_def = ToolDef(
    type="function",
    function=FunctionDescription(
        name="web_search",
        description="Search the web for up-to-date information. Search results contain the page title, href and a fragment of content",
        parameters=Parameters(
            type="object",
            properties={
                "query": ParameterProperty(
                    type="string",
                    description='Search query string. Supports advanced operators like double quotes for exact phrases, "+" to include terms, "-" to exclude terms, and domain-specific filters.',
                ),
                "scrape_pages": ParameterProperty(
                    type="boolean",
                    description="When true, automatically scrapes full content of result pages (using truncation rules from scrape_web_page). When false, returns only metadata and content fragments.",
                ),
                "max_results": ParameterProperty(
                    type="number",
                    description="Maximum number of results to return. Use to balance completeness vs response size. Defaults to 10 if unspecified.",
                ),
                "truncate_content": ParameterProperty(
                    type="number",
                    description="Maximum character length for content when `scrape_pages` is enabled. Content exceeding this limit will be truncated. Default is 10000.",
                ),
            },
            required=["query", "scrape_pages"],
        ),
    ),
)


def handle_net_tool(
    query: str, scrape_pages: bool, max_results: int = 10, truncate_content: int = 10000
) -> ToolResult:
    print(
        f"Called web_search with query: {query} and scrape_pages: {scrape_pages}",
        flush=True,
    )
    ts = time.time()
    results = DDGS().text(query, max_results=max_results)
    if not scrape_pages:
        res = ToolResult(
            status="success", stdout=json.dumps(results), stderr="", returncode=0
        )
    else:
        new_results = [
            {
                "title": res["title"],
                "href": res["href"],
                "content": handle_scrape_tool(
                    res["href"], truncate_content, source="web_search"
                ).model_dump_json(),
            }
            for res in results
        ]
        res = ToolResult(
            status="success", stdout=json.dumps(new_results), stderr="", returncode=0
        )
    print(f"Taken {time.time() - ts:.1f} seconds to complete.")
    if res.status == "error":
        if len(res.stderr) > 150:
            print(f"Tool evaluation led to error: {res.stderr[:100]} ... {res.stderr[-50:]}")
        else:
            print(f"Tool evaluation led to error: {res.stderr}")
    return res


net_tool = Tool(**{"definition": net_tool_def, "handler": handle_net_tool})
