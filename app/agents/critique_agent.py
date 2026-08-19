from app.agents.base import invoke_with_timeout
from app.core.config import settings
from app.core.logging import get_logger
from app.models.agent_io import CritiqueOutput
from app.prompts.critique_prompt import CRITIQUE_PROMPT
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


def run_critique_agent(query: str, draft: str, sources: list[dict]) -> CritiqueOutput:
    """Execute the critique agent workflow.

    Evaluates the research draft against the original query and the provided sources
    for completeness, structure, and strict factual grounding.

    Args:
        query: The user's input research query.
        draft: The research report draft.
        sources: The list of sources used to write the draft.

    Returns:
        A CritiqueOutput model containing the approval status and critique feedback.

    Raises:
        ValueError: If draft is empty/whitespace-only or if sources list is empty.
    """
    if not draft or not draft.strip():
        raise ValueError("Research draft cannot be empty or whitespace-only.")

    if not sources:
        raise ValueError("Sources list cannot be empty.")

    sources_context = format_sources(sources)

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(CritiqueOutput)
    chain = CRITIQUE_PROMPT | structured_llm

    logger.info("Critique agent: evaluating research draft.")

    # Wrap structured LLM call in generic timeout helper
    critique_output: CritiqueOutput = invoke_with_timeout(
        lambda: chain.invoke(
            {
                "query": query,
                "draft": draft,
                "sources_context": sources_context,
            }
        ),
        timeout_seconds=settings.AGENT_TIMEOUT,
    )

    feedback_preview = (
        critique_output.feedback[:60] + "..."
        if len(critique_output.feedback) > 60
        else critique_output.feedback
    )
    logger.info(
        f"Critique agent complete. Approved: {critique_output.approved}. "
        f"Feedback Preview: '{feedback_preview}'"
    )

    return critique_output
