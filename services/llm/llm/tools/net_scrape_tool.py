import time

import requests
from html2text import HTML2Text
from langchain_core.tools import StructuredTool
from readability import Document


def _get_clean_page_content(url: str, n_truncate: int):
    try:
        # 1. Fetch page
        headers = {'User-Agent': 'Mozilla/5.0'}
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
        return f'Failed to retrieve content: {str(e)}'


def scrape_web_page(
    href: str, truncate_content: int = 10000, source: str = 'inference'
) -> str:
    if source == 'inference':
        print(f'Called scrape_web_page with href: {href}', flush=True)
    ts = time.time()
    result = _get_clean_page_content(href, truncate_content)

    if source == 'inference':
        print(f'Taken {time.time() - ts:.1f} seconds to complete.')
    return result


TOOLNAME = 'scrape_web_page'

scrape_tool = StructuredTool.from_function(
    name=TOOLNAME,
    func=scrape_web_page,
    description='Fetches and returns the full text content of a web page from a specified URL. Optionally truncates the content to a specified character length. \
        Arguments:\
            - href: string - The full URL (including protocol) of the web page to scrape content from.\
            - truncate_content: int - optional maximum character length for the returned content. Content exceeding this limit will be truncated. 10000 characters by default.',
)
