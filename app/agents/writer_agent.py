from app.agents.base import invoke_with_timeout
from app.core.config import settings
from app.core.logging import get_logger
from app.prompts.writer_prompt import REVISION_PROMPT, WRITER_PROMPT
from app.services.llm_service import get_llm

logger = get_logger(__name__)


def format_sources(sources: list[dict]) -> str:
    """Formats a list of sources into a human-readable text context for the prompt."""
    formatted = []
    for idx, src in enumerate(sources):
        url = src.get("url", "No URL")
        title = src.get("title", "Untitled Source")
        content = src.get("relevant_content", "")
        formatted.append(
            f"[{idx + 1}] Title: {title}\nURL: {url}\nContent: {content}\n"
        )
    return "\n".join(formatted)


def run_writer_agent(
    query: str,
    sources: list[dict],
    critique_feedback: str | None = None,
    previous_draft: str | None = None,
) -> str:
    """Execute the writer agent workflow.

    Synthesizes the provided processed sources into a well-structured research report draft (first-draft mode),
    or refines an existing draft according to specific critique feedback (revision mode).

    Args:
        query: The user's input research query.
        sources: A list of relevant sources processed by the reader agent.
        critique_feedback: Optional review critique feedback. Must be paired with previous_draft.
        previous_draft: Optional previous draft text. Must be paired with critique_feedback.

    Returns:
        The written or revised draft as a plain text string.

    Raises:
        ValueError: If sources list is empty, or if exactly one of critique_feedback/previous_draft is provided.
    """
    if not sources:
        raise ValueError("Sources list cannot be empty.")

    # Validate mode parameter consistency
    has_feedback = critique_feedback is not None
    has_draft = previous_draft is not None
    if has_feedback != has_draft:
        raise ValueError(
            "Both critique_feedback and previous_draft must be provided for a revision, "
            "or both must be absent for a first draft."
        )

    # Format sources context
    sources_context = format_sources(sources)

    # Resolve mode and prompt chain
    is_revision = has_feedback and has_draft
    llm = get_llm(temperature=0.4)

    if is_revision:
        logger.info("Writer agent running in REVISION mode.")
        prompt_input = {
            "query": query,
            "critique_feedback": critique_feedback,
            "previous_draft": previous_draft,
            "sources_context": sources_context,
        }
        chain = REVISION_PROMPT | llm
    else:
        logger.info("Writer agent running in FIRST-DRAFT mode.")
        prompt_input = {"query": query, "sources_context": sources_context}
        chain = WRITER_PROMPT | llm

    # Execute LLM call with generic timeout wrapper
    # ChatGroq returns an AIMessage, extract the content property.
    response = invoke_with_timeout(
        lambda: chain.invoke(prompt_input), timeout_seconds=settings.AGENT_TIMEOUT
    )

    draft = getattr(response, "content", str(response))

    logger.info(
        f"Writer agent complete. Generated draft length: {len(draft)} characters."
    )
    return draft
