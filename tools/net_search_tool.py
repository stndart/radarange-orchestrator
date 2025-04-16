from .tool_annotation import ToolResult, ToolType, Tool

net_tool_def: ToolType = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for up-to-date information. Search results contain the page title, href and a fragment of content",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string. Supports advanced operators like double quotes for exact phrases, \"+\" to include terms, \"-\" to exclude terms, and domain-specific filters."
                },
                "scrape_pages": {
                    "type": "boolean",
                    "description": "When true, automatically scrapes full content of result pages (using truncation rules from scrape_web_page). When false, returns only metadata and content fragments."
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of results to return. Use to balance completeness vs response size. Defaults to 10 if unspecified."
                },
                "truncate_content": {
                    "type": "number",
                    "description": "Maximum character length for content when `scrape_pages` is enabled. Content exceeding this limit will be truncated. Default is 10000."
                }
            },
            "required": ["query", "scrape_pages"]
        }
    }
}

from duckduckgo_search import DDGS
from .net_scrape_tool import handle_scrape_tool
import json

def handle_net_tool(query: str, scrape_pages: bool, max_results: int = 10, truncate_content: int = 10000) -> ToolResult:
    print(f"Called web_search with query: {query} and scrape_pages: {scrape_pages}", flush=True)
    results = DDGS().text(query, max_results=max_results)
    if not scrape_pages:
        return {"status": "success", "stdout": json.dumps(results), "stderr": "", "returncode": 0}
    else:
        new_results = [{
            "title": res["title"],
            "href": res["href"],
            "content": handle_scrape_tool(res["href"], truncate_content, source='web_search')
        } for res in results]
        return {"status": "success", "stdout": json.dumps(new_results), "stderr": "", "returncode": 0}

net_tool: Tool = {
    "definition": net_tool_def,
    "handler": handle_net_tool
}