"""
helper functions
"""

from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from fairsenseai.runtime import get_runtime
import pandas as pd
import re


def row_to_text(row, mode="risk") -> str:
    """
    Converts a DataFrame row to a formatted text string based on the specified mode.

    Parameters
    ----------
    row : pandas.Series
        A row from a DataFrame containing risk or AI RMF information
    mode : str, optional
        The conversion mode - either 'risk' or 'ai_rmf', by default 'risk'

    Returns
    -------
    str
        Formatted text string combining key fields from the row

    Examples
    --------
    >>> risk_row = pd.Series({
    ...     'RiskID': 123,
    ...     'RiskCategory': 'Privacy',
    ...     'RiskDescription': 'Unauthorized data access'
    ... })
    >>> row_to_text(risk_row, mode='risk')
    'Risk Category: Privacy | Risk Description: Unauthorized data access'

    >>> rmf_row = pd.Series({
    ...     'section_name': 'Security',
    ...     'short_description': 'Data protection measures'
    ... })
    >>> row_to_text(rmf_row, mode='ai_rmf')
    'Short Description: Data protection measures | About: Data protection measures'
    """

    if mode == "risk":
        return f"Risk Category: {row['RiskCategory']} | Risk Description: {row['RiskDescription']}"
    if mode == "ai_rmf":
        return f"Short Description: {row['short_description']} | About: {row['short_description']}"


def post_process_response(response: str, use_summarizer: Optional[bool] = True) -> str:
    """
    Post-processes the response by optionally summarizing if the text is long
    and returning formatted HTML.

    Parameters
    ----------
    response
        The generated response text.
    use_summarizer
        Whether to use the summarizer to condense the response.

    Returns
    -------
    str
        The post-processed response with HTML formatting.
    """
    fairsense_runtime = get_runtime()

    cleaned_response = " ".join(response.split())

    # Only summarize if the checkbox is enabled and the text is long
    if use_summarizer and len(cleaned_response.split()) > 50:
        try:
            summary = fairsense_runtime.summarizer(
                cleaned_response, max_length=200, min_length=50, do_sample=False
            )
            cleaned_response = summary[0]["summary_text"]
        except Exception as e:
            cleaned_response = f"Error during summarization: {e}\nOriginal response: {cleaned_response}"

    # Clean up text into sentences
    sentences = [sentence.strip() for sentence in cleaned_response.split(".")]
    cleaned_response = ". ".join(sentences).strip() + (
        "." if not cleaned_response.endswith(".") else ""
    )
    return f"<strong>Here is the analysis:</strong> {cleaned_response}"


def highlight_bias(text: str, bias_words: List[str]) -> str:
    """
    Highlights bias words in the text with inline HTML styling.

    Parameters
    ----------
    text
        The input text to highlight.
    bias_words
        A list of bias words to highlight.

    Returns
    -------
    str
        The text with bias words highlighted in HTML.
    """
    if not bias_words:
        return f"<div>{text}</div>"
    for word in bias_words:
        text = text.replace(
            word, f"<span style='color: red; font-weight: bold;'>{word}</span>"
        )
    return f"<div>{text}</div>"


def style_risks(df: pd.DataFrame) -> str:
    """
    Generates HTML output highlighting specified risk IDs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing risk information with columns 'RiskID' and 'RiskDescription'

    Returns
    -------
    str
        HTML formatted string with styled risk entries

    Examples
    --------
    >>> risks_df = pd.DataFrame({
    ...     'RiskID': [1, 2, 3],
    ...     'RiskDescription': ['Privacy risk', 'Security risk', 'Bias risk']
    ... })
    >>> style_risks(risks_df)
    '<ul><li style="color:red; font-weight:bold;">Risk #1: Privacy risk</li>
    <li style="color:red; font-weight:bold;">Risk #3: Bias risk</li></ul>'
    """

    styles = """
    <style>
        .risk-container {
            max-width: 1200px;
            margin: 20px auto;
            font-family: Arial, sans-serif;
        }
        .risk-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .risk-title {
            color: #dc2626;
            font-size: 1.25rem;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .risk-category {
            color: #991b1b;
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 8px;
        }
        .risk-description {
            color: #4b5563;
            line-height: 1.5;
            margin-top: 8px;
        }
        .warning-icon {
            color: #dc2626;
            font-size: 1.25rem;
        }
    </style>
    """

    # Start HTML container
    html_output = styles + '<div class="risk-container">'

    # Add each risk entry
    for _, row in df.iterrows():
        risk_id = str(row["MIT Risk ID"])
        risk_category = str(row["MIT Risk Category"])
        risk_desc = str(row["MIT Risk Description"])

        # Create card for each risk
        html_output += f"""
        <div class="risk-card">
            <div class="risk-title">
                <span class="warning-icon">⚠️</span>
                <span>MIT Risk #{risk_id}</span>
            </div>
            <div class="risk-category">
                Category: {risk_category}
            </div>
            <div class="risk-description">
                {risk_desc}
            </div>
        </div>
        """

    # Close container
    html_output += "</div>"

    return html_output


def preprocess_image(image: Image) -> Image:
    """
    Preprocesses the image for OCR and captioning.

    Parameters
    ----------
    image
        The input image to preprocess.

    Returns
    -------
    Image
        The preprocessed image for OCR and captioning.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB))
