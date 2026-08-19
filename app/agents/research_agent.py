from app.agents.base import invoke_with_timeout
from app.core.config import settings
from app.core.logging import get_logger
from app.exceptions.custom_exceptions import ToolExecutionError
from app.models.agent_io import SearchQueries
from app.prompts.research_prompt import RESEARCH_PROMPT
from app.services.llm_service import get_llm
from app.tools.registry import TOOLS

logger = get_logger(__name__)


def run_research_agent(query: str, max_results_per_query: int = 5) -> list[dict]:
    """Execute the research agent workflow.

    Generates 2-3 targeted search queries using the Groq LLM, executes the searches
    via the tools registry, deduplicates resulting pages by URL keeping the highest-scoring
    version, and returns the aggregated list.

    Args:
        query: The user's input research question.
        max_results_per_query: The maximum number of results to fetch per search query.

    Returns:
        A list of source result dictionaries.

    Raises:
        ToolExecutionError: If all search queries fail to execute.
    """
    # Build structured LLM chain
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(SearchQueries)
    chain = RESEARCH_PROMPT | structured_llm

    # Execute LLM call with generic timeout wrapper
    logger.info(f"Generating search queries for user query: '{query}'")
    search_queries: SearchQueries = invoke_with_timeout(
        lambda: chain.invoke({"question": query}), timeout_seconds=settings.AGENT_TIMEOUT
    )

    if not search_queries or not search_queries.queries:
        raise ToolExecutionError("LLM failed to generate any search queries.")

    logger.info(f"Generated search queries: {search_queries.queries}")

    search_web = TOOLS["web_search"]
    aggregated_results: list[dict] = []
    failed_queries_count = 0

    for idx, q in enumerate(search_queries.queries):
        try:
            logger.info(
                f"Executing search query {idx + 1}/{len(search_queries.queries)}: '{q}'"
            )
            results = search_web(q, max_results=max_results_per_query)
            aggregated_results.extend(results)
        except ToolExecutionError as exc:
            failed_queries_count += 1
            logger.warning(
                f"Search query '{q}' failed to execute: {str(exc)}. Skipping sub-query."
            )

    # If all generated queries failed to resolve, raise error since we have no data
    if failed_queries_count == len(search_queries.queries):
        raise ToolExecutionError("All generated search queries failed to execute.")

    # Deduplicate results by URL, preserving the one with the highest score
    deduplicated_sources: dict[str, dict] = {}
    for res in aggregated_results:
        url = res.get("url")
        if not url:
            continue
        score = res.get("score", 0.0)
        if url not in deduplicated_sources or score > deduplicated_sources[url].get(
            "score", 0.0
        ):
            deduplicated_sources[url] = res

    final_sources = list(deduplicated_sources.values())
    logger.info(
        f"Research agent run complete. Generated {len(search_queries.queries)} queries, "
        f"retrieved {len(final_sources)} deduplicated sources."
    )

    return final_sources
