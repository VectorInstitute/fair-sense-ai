from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from fairsenseai.runtime import get_runtime

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

    cleaned_response = ' '.join(response.split())

    # Only summarize if the checkbox is enabled and the text is long
    if use_summarizer and len(cleaned_response.split()) > 50:
        try:
            summary = fairsense_runtime.summarizer(
                cleaned_response,
                max_length=200,
                min_length=50,
                do_sample=False
            )
            cleaned_response = summary[0]['summary_text']
        except Exception as e:
            cleaned_response = f"Error during summarization: {e}\nOriginal response: {cleaned_response}"

    # Clean up text into sentences
    sentences = [sentence.strip() for sentence in cleaned_response.split('.')]
    cleaned_response = '. '.join(sentences).strip() + (
        '.' if not cleaned_response.endswith('.') else ''
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
            word,
            f"<span style='color: red; font-weight: bold;'>{word}</span>"
        )
    return f"<div>{text}</div>"

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