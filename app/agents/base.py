import concurrent.futures
from typing import Any, Callable

from app.core.config import settings
from app.core.logging import get_logger
from app.exceptions.custom_exceptions import AgentTimeoutError

# Standard logger setup pattern to be followed consistently by other agent files:
# from app.core.logging import get_logger
# logger = get_logger(__name__)
logger = get_logger(__name__)


def invoke_with_timeout(
    llm_call: Callable[[], Any], timeout_seconds: int | None = None
) -> Any:
    """Invokes an LLM or agent callable within a specified timeout limit.

    Runs the call inside a ThreadPoolExecutor.

    WARNING: Because this executes synchronously, running it directly inside an async
    FastAPI endpoint will block the event loop if the execution is not delegated.
    When wiring into async FastAPI route handlers (Phase 6/7), run this via:
    `asyncio.get_running_loop().run_in_executor(None, run_research_agent, ...)`
    or rewrite the agent calls using LangChain's async methods (`ainvoke`)
    wrapped in `asyncio.wait_for(...)`.

    Args:
        llm_call: A parameter-less callable containing the LLM invocation.
        timeout_seconds: The timeout limit in seconds.

    Returns:
        The result of the llm_call execution.

    Raises:
        AgentTimeoutError: If execution exceeds the specified timeout limit.
    """
    timeout = timeout_seconds if timeout_seconds is not None else settings.AGENT_TIMEOUT
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm_call)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            logger.error(f"LLM call timed out after {timeout} seconds.")
            raise AgentTimeoutError(
                message=f"Agent invocation timed out after {timeout} seconds.",
                status_code=504,
            ) from exc
