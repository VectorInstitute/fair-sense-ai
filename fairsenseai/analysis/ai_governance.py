"""
Tool to generate insights and recommendations on AI governance and safety topics.
"""

from typing import Callable

from fairsenseai.runtime import generate_response_with_model
from fairsenseai.utils.helper import post_process_response

def ai_governance_response(
    prompt: str,
    use_summarizer: bool = True,  # <-- Summarizer toggle
    progress: Callable[[float, str], None] = None
) -> str:
    """
    Generates insights and recommendations on AI governance and safety topics.

    Parameters
    ----------
    prompt
        The input topic or question on AI governance and safety.
    use_summarizer
        Whether to use the summarizer to condense the response.
    progress
        A callback function to report progress.

    Returns
    -------
    str
        The generated response with insights and recommendations on AI Governance and Safety.

    Example
    -------
    >>> ai_governance_response("Environment Impact of AI")
    """
    response = generate_response_with_model(
        f"Provide insights and recommendations on the following AI governance and safety topic:\n\n{prompt}",
        progress=progress
    )
    # Use summarizer toggle
    return post_process_response(response, use_summarizer=use_summarizer)