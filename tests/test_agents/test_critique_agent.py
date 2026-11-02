from unittest.mock import MagicMock, patch

import pytest

from app.agents.critique_agent import format_sources, run_critique_agent
from app.exceptions.custom_exceptions import AgentTimeoutError
from app.models.agent_io import CritiqueOutput


@patch("app.agents.critique_agent.get_llm")
def test_run_critique_agent_approved(mock_get_llm):
    """Verify that critique agent successfully approves a draft and returns CritiqueOutput."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = CritiqueOutput(
        approved=True, feedback="Looks good."
    )
    mock_structured.return_value = CritiqueOutput(approved=True, feedback="Looks good.")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    sources = [
        {
            "url": "https://source.com",
            "title": "Source 1",
            "relevant_content": "Content",
        }
    ]
    result = run_critique_agent("FastAPI query", "My draft content.", sources)

    assert result.approved is True
    assert result.feedback == "Looks good."
    assert mock_llm.with_structured_output.call_count == 1
    assert mock_structured.call_count == 1


@patch("app.agents.critique_agent.get_llm")
def test_run_critique_agent_rejected(mock_get_llm):
    """Verify critique agent rejects draft and passes critique feedback unmodified."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    feedback_text = "The claim in paragraph 2 is not supported by any source."
    mock_structured.invoke.return_value = CritiqueOutput(
        approved=False, feedback=feedback_text
    )
    mock_structured.return_value = CritiqueOutput(
        approved=False, feedback=feedback_text
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    sources = [
        {
            "url": "https://source.com",
            "title": "Source 1",
            "relevant_content": "Content",
        }
    ]
    result = run_critique_agent("FastAPI query", "My draft content.", sources)

    assert result.approved is False
    assert result.feedback == feedback_text
    assert mock_structured.call_count == 1


def test_run_critique_agent_empty_draft():
    """Verify that an empty or whitespace-only draft raises ValueError immediately."""
    with patch("app.agents.critique_agent.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        sources = [{"url": "https://source.com"}]

        # Case A: empty draft
        with pytest.raises(ValueError) as exc_info:
            run_critique_agent("Query", "", sources)
        assert "Research draft cannot be empty" in str(exc_info.value)

        # Case B: whitespace-only draft
        with pytest.raises(ValueError) as exc_info:
            run_critique_agent("Query", "    ", sources)
        assert "Research draft cannot be empty" in str(exc_info.value)

        assert mock_llm.with_structured_output.call_count == 0


@patch("app.agents.critique_agent.get_llm")
def test_run_critique_agent_empty_sources(mock_get_llm):
    """Verify that an empty sources list raises ValueError immediately before any LLM call."""
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    with pytest.raises(ValueError) as exc_info:
        run_critique_agent("Query", "Draft content", [])

    assert "Sources list cannot be empty" in str(exc_info.value)
    assert mock_llm.with_structured_output.call_count == 0


@patch("app.agents.critique_agent.get_llm")
def test_run_critique_agent_timeout_propagation(mock_get_llm):
    """Verify AgentTimeoutError raised from timed-out LLM call propagates uncaught."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.side_effect = AgentTimeoutError("LLM call timed out")
    mock_structured.invoke.side_effect = AgentTimeoutError("LLM call timed out")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    sources = [{"url": "https://source.com"}]

    with pytest.raises(AgentTimeoutError) as exc_info:
        run_critique_agent("Query", "Draft content", sources)

    assert "LLM call timed out" in str(exc_info.value)


def test_format_sources_truncation():
    """Verify format_sources truncates relevant_content to max_chars_per_source (default 800)."""
    long_content = "X" * 1500
    sources = [
        {
            "url": "https://source1.com",
            "title": "Long Source",
            "relevant_content": long_content,
        },
        {
            "url": "https://source2.com",
            "title": "Short Source",
            "relevant_content": "Short text",
        },
    ]

    formatted = format_sources(sources)
    assert f"Content: {'X' * 800}\n" in formatted
    assert "Content: Short text\n" in formatted
    assert ("X" * 801) not in formatted

    # Test custom max_chars_per_source
    custom_formatted = format_sources(sources, max_chars_per_source=500)
    assert f"Content: {'X' * 500}\n" in custom_formatted
    assert ("X" * 501) not in custom_formatted


@patch("app.agents.critique_agent.get_llm")
def test_run_critique_agent_payload_truncation(mock_get_llm):
    """Verify run_critique_agent truncates source content before sending to prompt chain."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = CritiqueOutput(
        approved=True, feedback="Valid"
    )
    mock_structured.return_value = CritiqueOutput(approved=True, feedback="Valid")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    long_content = "A" * 2000
    sources = [
        {
            "url": "https://example.com",
            "title": "Article",
            "relevant_content": long_content,
        }
    ]

    result = run_critique_agent("Query", "Draft body", sources)
    assert result.approved is True

    # Check the prompt input passed to chain
    assert mock_structured.call_count == 1
    args, kwargs = mock_structured.call_args
    prompt_str = str(args[0])
    assert "A" * 800 in prompt_str
    assert "A" * 801 not in prompt_str
