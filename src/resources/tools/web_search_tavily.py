"""Web search tool making use of the Tavily Client & API"""

from langchain_core.tools import tool, InjectedToolArg
from typing_extensions import Annotated, Literal
from src.resources.external_modules.tavily_manager import TavilyWebSearchManager

_tavily_manager: TavilyWebSearchManager | None = None


def set_tavily_manager_for_tools(manager):
    """Configure the Tavily manager instance to be reused by tools."""
    global _tavily_manager
    _tavily_manager = manager


@tool(parse_docstring=True)
async def tavily_web_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    if not _tavily_manager:
        raise RuntimeError("No Tavily manager configured for the web search tool")

    search_results = _tavily_manager.tavily_search_multiple(
        [query], max_results=max_results, topic=topic, include_raw_content=True
    )
    unique_results = _tavily_manager.deduplicate_search_results(search_results)
    summarized_results = await _tavily_manager.process_search_results(unique_results)
    return _tavily_manager.format_search_output(summarized_results)
