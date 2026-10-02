from langchain_core.tools import tool
from tavily import TavilyClient

from src.config import settings

tavily = TavilyClient(api_key=settings.tavily_key)

@tool
def search_tool(query: str):
    """Search the web for information."""
    response = tavily.search(
        query=query,
        search_depth='advanced',
        max_results=3
    )

    results = []

    for result in response['results']:
        results.append(
            f'Title: {result['title']}\n'
            f'URL: {result['url']}\n'
            f'Content: {result['content'][:2000]}'
        )

    return '\n\n'.join(results)