import json
import time
from typing import Any, Optional

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from .net_scrape_tool import scrape_web_page

_good_instances = [
    'https://4get.dcs0.hu',
    'https://4get.ch',
    'https://4get.edmateo.site',
    'https://search.mint.lgbt',
    'https://4get.aishiteiru.moe',
    'https://4get.thebunny.zone',
    'https://4.nboeck.de',
    'https://search.yonderly.org',
    'https://4get.kuuro.net',
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


def web_search(
    query: str, scrape_pages: bool, max_results: int = 10, truncate_content: int = 10000
) -> list[dict]:
    print(
        f'Called web_search with query: {query} and scrape_pages: {scrape_pages}',
        flush=True,
    )
    ts = time.time()

    results = _extract_links(_raw(_instance, query))

    print(f'Taken {time.time() - ts:.1f} seconds to complete search.')
    print(f'web search: {results}')
    if scrape_pages:
        ts = time.time()
        results = [
            {
                'title': res.title,
                'url': res.url,
                'content': scrape_web_page(res.url, truncate_content, source='web_search'),
            }
            for res in results
        ]
        print(
            f'Taken {time.time() - ts:.1f} seconds to complete scraping pages with mean scrape time {(time.time() - ts) / len(results):.1f} seconds.'
        )
    else:
        results = [
            link.model_dump_json() for link in _extract_links(_raw(_instance, query))
        ]

    return results


TOOLNAME = 'web_search'

net_tool = StructuredTool.from_function(
    name=TOOLNAME,
    func=web_search,
    description='Search the web for up-to-date information. Search results contain the page title, url and a fragment of content. \
        Arguments:\
            - query: string\
            - scrape_pages: bool - whether to scrapes content of result page. Default is False.\
            - max_results: int - number of search results to return. Default is 10.\
            - truncate_content: int - maximum character length of the content when `scrape_pages` is enabled. Default is 10000.',
)
