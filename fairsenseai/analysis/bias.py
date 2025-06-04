"""
Functions for analyzing biases in text and images.
"""

import base64
import logging
import os
import time
from io import BytesIO
from typing import List, TextIO, Tuple, Optional

from PIL import Image
import pandas as pd
import pytesseract
import torch
import gradio as gr
import re

logger = logging.getLogger(__name__)



from fairsenseai.runtime import generate_response_with_model, get_runtime
from fairsenseai.utils.helper import highlight_bias, post_process_response, preprocess_image

def analyze_text_for_bias(
    text_input: str,
    use_summarizer: bool,
    progress: gr.Progress = gr.Progress()
) -> Tuple[str, str, int]:
    """
    Analyzes a given text for bias and provides a detailed analysis.

    Parameters
    ----------
    text_input
        The input text to analyze for bias.
    use_summarizer
        Whether to use the summarizer to condense the response.
    progress
        A callback function to report progress.

    Returns
    -------
    Tuple[str, str, int]
        A tuple containing the highlighted text with bias words, the detailed analysis and the bias score percentage.

    Example
    -------
    >>> highlighted, analysis, score = analyze_text_for_bias("This text may contain bias.", use_summarizer=True)
    """

    progress(0, "Initializing analysis...")

    try:
        time.sleep(0.2)  # Simulate delay for initializing
        progress(0.1, "Preparing analysis...")

        prompt = (
            f"Analyze the following text for bias. Be concise, focusing only on relevant details. "
            f"Mention specific phrases or language that contribute to bias, and describe the tone of the text. "
            f"Mention who is the targeted group (if any). "
            f"Provide your response as a clear and concise paragraph. If no bias is found, state that the text appears unbiased. "
            f"Also provide a score from 0 to 100 indicating the level of bias as a percentage, where 0% is unbiased and 100% is highly biased.\n\n"
            f"Response format:\nScore: <number>\nExplanation: <paragraph>\n\n"
            f"Text: \"{text_input}\""
        )

        progress(0.3, "Generating response...")
        response = generate_response_with_model(
            prompt,
            progress=lambda x, desc="": progress(0.3 + x * 0.4, desc),
        )
        
        score_match = re.search(r"Score:\s*(\d{1,3})", response)
        bias_score = -1
        if score_match:
            try:
                score_value = int(score_match.group(1))
                if 0 <= score_value <= 100:
                    bias_score = score_value
                else:
                    logger.warning(f"Bias score out of range (0–100): {score_value}")
            except ValueError:
                logger.info(f"Unable to parse bias score: {score_match.group(1)}")
        else:
            logger.info("Bias score not found in the response")
        # Extracting the explanation part
        explanation_match = re.search(r"Explanation:\s*(.*)", response, re.DOTALL)
        explanation_text = explanation_match.group(1).strip() if explanation_match else response
        # Remove the "Score" and score output from the explanation
        explanation_text = re.sub(r"Score:\s*\d{1,3}", "", explanation_text).strip()

        
        progress(0.7, "Post-processing response...")
        processed_response = post_process_response(explanation_text, use_summarizer=use_summarizer)

        progress(0.9, "Highlighting text bias...")
        bias_section = response.split("Biased words:")[-1].strip() if "Biased words:" in response else ""
        biased_words = [word.strip() for word in bias_section.split(",")] if bias_section else []
        highlighted_text = highlight_bias(text_input, biased_words)

        progress(1.0, "Analysis complete.")
        return highlighted_text, processed_response, bias_score
    except Exception as e:
        progress(1.0, "Analysis failed.")
        return f"Error: {e}", "", -1  # -1 as fallback for bias score
    
def analyze_image_for_bias(
    image: Image,
    use_summarizer: bool,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Analyzes an image for bias by extracting text using OCR and by generating image caption.

    Parameters
    ----------
    image
        The input image to analyze for bias.
    use_summarizer
        Whether to use the summarizer to condense the response.
    progress
        A callback function to report progress.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the highlighted text with bias words and the detailed analysis.

    Example
    -------
    >>> image = Image.open("example.jpg")
    >>> highlighted, analysis = analyze_image_for_bias(image, use_summarizer=True)
    """
    fair_sense_runtime = get_runtime()

    progress(0, "Initializing image analysis...")

    try:
        time.sleep(0.1)  # Simulate delay
        progress(0.1, "Processing image...")

        # Ensure RGB
        image = image.convert("RGB")

        # Prepare inputs for BLIP
        inputs = fair_sense_runtime.blip_processor(images=image, return_tensors="pt").to(fair_sense_runtime.device)

        progress(0.2, "Extracting text from image...")
        preprocessed_image = preprocess_image(image)
        extracted_text = pytesseract.image_to_string(preprocessed_image)

        progress(0.3, "Generating caption...")
        with torch.no_grad():
            caption_ids = fair_sense_runtime.blip_model.generate(
                **inputs,
                max_length=300,
                num_beams=5,
                temperature=0.7
            )
        caption_text = fair_sense_runtime.blip_processor.tokenizer.decode(
            caption_ids[0],
            skip_special_tokens=True
        ).strip()

        combined_text = f"{caption_text}. {extracted_text}"

        progress(0.6, "Analyzing combined text for bias...")
        prompt = (
            f"Analyze the following image-related text for bias, mockery, misinformation, disinformation, "
            f"or satire. Mention any targeted group if found. "
            f"If no bias is found, state that the image appears unbiased.\n\n"
            f"Text:\n\"{combined_text}\""
        )
        response = generate_response_with_model(
            prompt,
            progress=lambda x, desc="": progress(0.6 + x * 0.3, desc)
        )

        progress(0.9, "Post-processing response...")
        processed_response = post_process_response(response, use_summarizer=use_summarizer)

        # Extract any biased words
        bias_section = response.split("Biased words:")[-1].strip() if "Biased words:" in response else ""
        biased_words = [word.strip() for word in bias_section.split(",")] if bias_section else []
        highlighted_text = highlight_bias(caption_text, biased_words) # Displaying only caption_text instead of combined_text for clean output

        progress(1.0, "Analysis complete.")
        return highlighted_text, processed_response
    except Exception as e:
        progress(1.0, f"Analysis failed: {e}")
        return f"Error: {e}", ""

    
def analyze_text_csv(
    file: TextIO,
    use_summarizer: bool,  # <--- Summarizer toggle for CSV
    output_filename: Optional[str] = "analysis_results.csv"
) -> str:
    """
    Analyzes a CSV file containing multiple text entries for bias.

    Parameters
    ----------
    file
        The input CSV file containing text data.
    use_summarizer
        Whether to use the summarizer to condense the response.
    output_filename
        The filename to save the analysis results.

    Returns
    -------    
    str
        The HTML table containing results of batch analysis. 

    Examples
    --------
    >>> csv_file = open("data.csv", mode='r', newline='', encoding='utf-8')
    >>> results_table_html = analyze_text_csv(csv_file, use_summarizer=True)
    """     

    fair_sense_runtime = get_runtime()

    try:
        df = pd.read_csv(file.name)
        if "text" not in df.columns:
            return "Error: The CSV file must contain a 'text' column."

        results = []
        for i, text in enumerate(df["text"]):
            try:
                highlighted_text, analysis, score = analyze_text_for_bias(text, use_summarizer=use_summarizer)
                results.append({
                    "row_index": i + 1,
                    "text": highlighted_text,
                    "analysis": analysis,
                    "bias_score_percentage": f"{score}%" if score != -1 else "N/A"
                })
            except Exception as e:
                results.append({
                    "row_index": i + 1,
                    "text": "Error",
                    "analysis": str(e),
                    "bias_score_percentage": -1
                })

        result_df = pd.DataFrame(results)
        html_table = result_df.to_html(escape=False)  # escape=False to render HTML in cells
        save_path = fair_sense_runtime.save_results_to_csv(result_df, output_filename)
        return html_table
    except Exception as e:
        return f"Error processing CSV: {e}"
    
def analyze_images_batch(
    images: List[str],
    use_summarizer: bool,  # <--- Summarizer toggle for images
    output_filename: Optional[str] = "image_analysis_results.csv"
) -> str:
    """
    Analyzes a batch of images for bias.

    Parameters
    ----------
    images
        The list of images to analyze for bias.
    use_summarizer
        Whether to use the summarizer to condense the response.
    output_filename
        The filename to save the analysis results.

    Returns
    -------    
    str
        The HTML table containing results of batch analysis.

    Example
    -------
    >>> results_table_html = analyze_images_batch(["image1.jpg", "image2.png"], use_summarizer=True)
    """
    fair_sense_runtime = get_runtime()

    try:
        results = []
        for i, image_path in enumerate(images):
            try:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                image = Image.open(image_path)
                highlighted_caption, analysis = analyze_image_for_bias(
                    image,
                    use_summarizer=use_summarizer
                )

                logging.info(f"Processing Image: {image_path}")

                # Convert image to base64 for HTML display
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_html = f'<img src="data:image/png;base64,{img_str}" width="200"/>'

                results.append({
                    "image_index": i + 1,
                    "image": img_html,
                    "analysis": analysis
                })
            except Exception as e:
                results.append({
                    "image_index": i + 1,
                    "image": "Error",
                    "analysis": str(e)
                })

        result_df = pd.DataFrame(results)
        html_table = result_df.to_html(escape=False)
        save_path = fair_sense_runtime.save_results_to_csv(result_df, output_filename)
        return html_table
    except Exception as e:
        return f"Error processing images: {e}"