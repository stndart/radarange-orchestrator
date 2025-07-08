import json
import time
from typing import Any, Optional

import requests
from pydantic import BaseModel

from ..types.tools import (
    FunctionDescription,
    ParameterProperty,
    Parameters,
    Tool,
    ToolDef,
    ToolResult,
)
from .net_scrape_tool import handle_scrape_tool

net_tool_def = ToolDef(
    type='function',
    function=FunctionDescription(
        name='web_search',
        description='Search the web for up-to-date information. Search results contain the page title, href and a fragment of content',
        parameters=Parameters(
            type='object',
            properties={
                'query': ParameterProperty(
                    type='string',
                    description='Search query string. Supports advanced operators like double quotes for exact phrases, "+" to include terms, "-" to exclude terms, and domain-specific filters.',
                ),
                'scrape_pages': ParameterProperty(
                    type='boolean',
                    description='When true, automatically scrapes full content of result pages (using truncation rules from scrape_web_page). When false, returns only metadata and content fragments.',
                ),
                'max_results': ParameterProperty(
                    type='number',
                    description='Maximum number of results to return. Use to balance completeness vs response size. Defaults to 10 if unspecified.',
                ),
                'truncate_content': ParameterProperty(
                    type='number',
                    description='Maximum character length for content when `scrape_pages` is enabled. Content exceeding this limit will be truncated. Default is 10000.',
                ),
            },
            required=['query', 'scrape_pages'],
        ),
    ),
)

_good_instances = [
    "https://4get.dcs0.hu",
    "https://4get.ch",
    "https://4get.edmateo.site",
    "https://search.mint.lgbt",
    "https://4get.aishiteiru.moe",
    "https://4get.thebunny.zone",
    "https://4.nboeck.de",
    "https://search.yonderly.org",
    "https://4get.kuuro.net",
]

_instance = _good_instances[0]

def _raw(inst: str, prompt: str) -> Optional[dict[str, Any]]:
    try:
        r = requests.get(f'{inst}/api/v1/web?s={prompt}')
    except Exception:
        return None
    return json.loads(r.content.decode())


def _extract_answer(query: dict[str, Any]) -> Optional[dict[str, Any]]:
    assert query['status'] == 'ok'
    return query.get('answer', None)


class SearchResult(BaseModel):
    title: str
    url: str
    description: str


def _extract_links(query: dict[str, Any]) -> list[SearchResult]:
    assert query['status'] == 'ok'

    results: list[SearchResult] = []
    for item in query['web']:
        results.append(
            SearchResult(
                title=item['title'],
                url=item['url'],
                description=item.get('description', ''),
            )
        )
    return results


def handle_net_tool(
    query: str, scrape_pages: bool, max_results: int = 10, truncate_content: int = 10000
) -> ToolResult:
    print(
        f'Called web_search with query: {query} and scrape_pages: {scrape_pages}',
        flush=True,
    )
    ts = time.time()
    
    try:
        results = _extract_links(_raw(_instance, query))
    except Exception as e:
        res = ToolResult(
            status='error', stderr=str(e), returncode=-1
        )
    else:
        print(f'Taken {time.time() - ts:.1f} seconds to complete search.')
        ts = time.time()
        if not scrape_pages:
            res = ToolResult(
                status='success', stdout=json.dumps(results), stderr='', returncode=0
            )
        else:
            new_results = [
                {
                    'title': res.title,
                    'url': res.url,
                    'content': handle_scrape_tool(
                        res.url, truncate_content, source='web_search'
                    ).model_dump_json(),
                }
                for res in results
            ]
            res = ToolResult(
                status='success', stdout=json.dumps(new_results), stderr='', returncode=0
            )
            print(
                f'Taken {time.time() - ts:.1f} seconds to complete scraping pages with mean scrape time {(time.time() - ts) / len(results):.1f} seconds.'
            )
    
    if res.status == 'error':
        if len(res.stderr) > 150:
            print(
                f'Tool evaluation led to error: {res.stderr[:100]} ... {res.stderr[-50:]}'
            )
        else:
            print(f'Tool evaluation led to error: {res.stderr}')
    return res


net_tool = Tool(**{'definition': net_tool_def, 'handler': handle_net_tool})
