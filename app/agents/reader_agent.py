from app.agents.base import invoke_with_timeout
from app.core.config import settings
from app.core.logging import get_logger
from app.exceptions.custom_exceptions import ToolExecutionError
from app.models.agent_io import CondensedContent
from app.prompts.reader_prompt import READER_PROMPT
from app.services.llm_service import get_llm
from app.tools.registry import TOOLS

logger = get_logger(__name__)


def run_reader_agent(
    query: str, sources: list[dict], max_sources: int = 5
) -> list[dict]:
    """Execute the reader agent workflow.

    Sorts the sources by descending score, takes the top `max_sources`, scrapes them,
    and runs a structured LLM call to extract and condense content relevant to the query.
    If LLM condensation fails/times out, falls back to raw scraped text truncated to 2,000 characters.
    If a source is marked `is_relevant=False`, it is skipped.
    Raises ToolExecutionError if all sources fail to scrape or yield zero relevant content.

    Args:
        query: The user's input research query.
        sources: A list of result dictionaries from the research agent.
        max_sources: The maximum number of sources to process.

    Returns:
        A list of processed source dictionaries containing 'url', 'title', and 'relevant_content'.

    Raises:
        ToolExecutionError: If all sources fail to scrape or yield relevant content.
    """
    # 1. Sort descending by score and slice to top max_sources
    sorted_sources = sorted(sources, key=lambda s: s.get("score", 0.0), reverse=True)
    top_sources = sorted_sources[:max_sources]

    scrape_url = TOOLS["scrape_url"]

    # Initialize LLM & structured chain
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(CondensedContent)
    chain = READER_PROMPT | structured_llm

    results: list[dict] = []
    attempted = len(top_sources)
    skipped = 0
    succeeded = 0

    for src in top_sources:
        url = src.get("url")
        title = src.get("title") or "Untitled Source"

        if not url:
            logger.warning("Encountered source with missing URL. Skipping.")
            skipped += 1
            continue

        # Fetch/Scrape URL
        logger.info(f"Scraping source: '{url}'")
        scrape_res = scrape_url(url)

        if not scrape_res.get("success"):
            logger.warning(f"Scraping failed for URL '{url}'. Skipping source.")
            skipped += 1
            continue

        scraped_text = scrape_res.get("text", "")

        # Invoke LLM Condensation wrapped in timeout
        try:
            logger.info(f"Condensing text for URL '{url}' using LLM")
            condensed: CondensedContent = invoke_with_timeout(
                lambda: chain.invoke({"query": query, "text": scraped_text}),
                timeout_seconds=settings.AGENT_TIMEOUT,
            )

            if not condensed.is_relevant:
                logger.info(
                    f"Source '{url}' deemed irrelevant by LLM. Skipping content."
                )
                skipped += 1
                continue

            relevant_content = condensed.relevant_content

        except Exception as exc:
            logger.warning(
                f"LLM condensation call failed or timed out for '{url}': {str(exc)}. "
                "Falling back to truncated raw scraped text."
            )
            # Fallback to truncated raw scraped text (first 2,000 characters)
            relevant_content = scraped_text[:2000]

        results.append(
            {"url": url, "title": title, "relevant_content": relevant_content}
        )
        succeeded += 1

    logger.info(
        f"Reader agent complete. Attempted: {attempted}, Skipped: {skipped}, Succeeded: {succeeded}"
    )

    if not results:
        raise ToolExecutionError(
            "All sources failed to scrape or yield relevant content."
        )

    return results
