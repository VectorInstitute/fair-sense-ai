import base64
import logging
import os
import time
from abc import ABCMeta, abstractmethod
from io import BytesIO
from typing import Callable, List, Optional, TextIO, Tuple

import cv2
import gradio as gr
import numpy as np
import ollama
import pandas as pd
import plotly.express as px
import pytesseract
import torch
from PIL import Image
from plotly.graph_objs import Figure
from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)

class FairsenseRuntime(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, allow_filesystem_access: bool = True):
        self.allow_filesystem_access = allow_filesystem_access
        if self.allow_filesystem_access:
            print("Starting FairseseRuntime with file system access.")
            self.default_directory = "bias-results"
            os.makedirs(self.default_directory, exist_ok=True)
        else:
            print("Starting FairseseRuntime without file system access.")


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blip_model_id = "Salesforce/blip-image-captioning-base"

        # Load Models
        print("Loading models...")
        try:
            # Image Captioning Models
            self.blip_processor = BlipProcessor.from_pretrained(self.blip_model_id)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_model_id).to(self.device)

            # Summarizer for post-processing
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=0 if torch.cuda.is_available() else -1
            )

            print("Models loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")

        # Expanded AI Safety Risks Data
        self.ai_safety_risks = [
            {"Risk": "Disinformation Spread", "Category": "Information Integrity", "Percentage": 20, "Severity": 8,
             "Likelihood": 7, "Impact": "High",
             "Description": "AI-generated content can spread false information rapidly.",
             "Mitigation": "Develop AI tools for fact-checking and verification."},
            {"Risk": "Algorithmic Bias", "Category": "Fairness and Bias", "Percentage": 18, "Severity": 7,
             "Likelihood": 8,
             "Impact": "High", "Description": "AI systems may perpetuate or amplify societal biases.",
             "Mitigation": "Implement fairness-aware algorithms and diverse datasets."},
            {"Risk": "Privacy Invasion", "Category": "Data Privacy", "Percentage": 15, "Severity": 6, "Likelihood": 6,
             "Impact": "Medium", "Description": "AI can infer personal information without consent.",
             "Mitigation": "Adopt privacy-preserving techniques like differential privacy."},
            {"Risk": "Lack of Transparency", "Category": "Explainability", "Percentage": 12, "Severity": 5,
             "Likelihood": 5,
             "Impact": "Medium", "Description": "Complex models can be opaque, making decisions hard to understand.",
             "Mitigation": "Use explainable AI methods to increase transparency."},
            {"Risk": "Security Vulnerabilities", "Category": "Robustness", "Percentage": 10, "Severity": 6,
             "Likelihood": 5,
             "Impact": "Medium", "Description": "AI systems may be susceptible to adversarial attacks.",
             "Mitigation": "Employ robust training methods and continuous monitoring."},
            {"Risk": "Job Displacement", "Category": "Economic Impact", "Percentage": 8, "Severity": 7, "Likelihood": 6,
             "Impact": "High", "Description": "Automation may lead to loss of jobs in certain sectors.",
             "Mitigation": "Promote reskilling and education programs."},
            {"Risk": "Ethical Dilemmas", "Category": "Ethics", "Percentage": 7, "Severity": 5, "Likelihood": 4,
             "Impact": "Medium", "Description": "AI may make decisions conflicting with human values.",
             "Mitigation": "Incorporate ethical guidelines into AI development."},
            {"Risk": "Autonomous Weapons", "Category": "Physical Safety", "Percentage": 5, "Severity": 9,
             "Likelihood": 3,
             "Impact": "Critical", "Description": "AI could be used in weapons without human oversight.",
             "Mitigation": "Establish international regulations and oversight."},
            {"Risk": "Environmental Impact", "Category": "Sustainability", "Percentage": 3, "Severity": 4,
             "Likelihood": 5,
             "Impact": "Low", "Description": "High energy consumption in AI training affects the environment.",
             "Mitigation": "Optimize models and use renewable energy sources."},
            {"Risk": "Misuse for Surveillance", "Category": "Human Rights", "Percentage": 2, "Severity": 8,
             "Likelihood": 2,
             "Impact": "High", "Description": "AI can be used for mass surveillance violating privacy rights.",
             "Mitigation": "Enforce laws protecting individual privacy."},
        ]

    def save_results_to_csv(self, df: pd.DataFrame, filename: str = "results.csv") -> Optional[str]:
        if not self.allow_filesystem_access:
            print("[ERROR] Not saving results to CSV because filesystem access is not allowed.")
            return None

        file_path = os.path.join(self.default_directory, filename)  # Combine directory and filename
        try:
            df.to_csv(file_path, index=False)  # Save the DataFrame as a CSV file
            return file_path  # Return the full file path for reference
        except Exception as e:
            return f"Error saving file: {e}"

    @abstractmethod
    def predict_with_text_model(self, prompt: str, progress: Callable[[float, str], None] = None) -> str:
        raise NotImplementedError


class FairsenseGPURuntime(FairsenseRuntime):
    def __init__(self, allow_filesystem_access: bool = True):
        super().__init__(allow_filesystem_access=allow_filesystem_access)
        self.text_model_hf_id = "unsloth/Llama-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_hf_id, use_fast=False)
        self.tokenizer.add_special_tokens({
            'eos_token': '</s>', 'bos_token': '<s>', 'unk_token': '<unk>', 'pad_token': '<pad>'
        })
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text_model = AutoModelForCausalLM.from_pretrained(self.text_model_hf_id).to(self.device).eval()
        self.text_model.resize_token_embeddings(len(self.tokenizer))

    def predict_with_text_model(self, prompt: str, progress: Callable[[float, str], None] = None) -> str:
        if progress:
            progress(0.1, "Tokenizing prompt...")
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024
        ).to(self.device)

        if progress:
            progress(0.3, "Generating response...")
        with torch.no_grad():
            outputs = self.text_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=1000,  # Adjusted max tokens
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                num_beams=3,
                temperature=0.7,
                repetition_penalty=1.2,
                early_stopping=True
            )

        if progress:
            progress(0.7, "Decoding output...")

        # Safeguard against index out of range
        start_index = inputs["input_ids"].shape[1]
        if len(outputs[0]) <= start_index:
            response = ""
        else:
            response = self.tokenizer.decode(
                outputs[0][start_index:], skip_special_tokens=True
            ).strip()

        if progress:
            progress(1.0, "Done")
        return response


class FairsenseCPURuntime(FairsenseRuntime):
    def __init__(self, allow_filesystem_access: bool = True):
        super().__init__(allow_filesystem_access=allow_filesystem_access)
        self.text_model_hf_id = "llama3.2"  # from ollama
        self.ollama_client = ollama.Client()

    def predict_with_text_model(self, prompt: str, progress: Callable[[float, str], None] = None) -> str:
        if progress:
            progress(0.1, "Preparing prompt...")

        if progress:
            progress(0.3, "Generating response...")

            # Generate response using Ollama
        response = self.ollama_client.chat(
            model=self.text_model_hf_id,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
        )
        if progress:
            progress(0.7, "Processing response...")

        # Extract the generated text from the response
        generated_text = response['message']['content']

        if progress:
            progress(1.0, "Done")
        return generated_text


FAIRSENSE_RUNTIME: Optional[FairsenseRuntime] = None
def initialize(allow_filesystem_access: bool = True) -> None:
    global FAIRSENSE_RUNTIME
    if torch.cuda.is_available():
        FAIRSENSE_RUNTIME = FairsenseGPURuntime(allow_filesystem_access=allow_filesystem_access)
    else:
        FAIRSENSE_RUNTIME = FairsenseCPURuntime(allow_filesystem_access=allow_filesystem_access)


# Helper Functions
def post_process_response(response: str) -> str:
    if FAIRSENSE_RUNTIME is None:
        initialize()

    cleaned_response = ' '.join(response.split())
    if len(cleaned_response.split()) > 50:
        try:
            summary = FAIRSENSE_RUNTIME.summarizer(cleaned_response, max_length=200, min_length=50, do_sample=False)
            cleaned_response = summary[0]['summary_text']
        except Exception as e:
            cleaned_response = f"Error during summarization: {e}\nOriginal response: {cleaned_response}"

    sentences = [sentence.strip() for sentence in cleaned_response.split('.')]
    cleaned_response = '. '.join(sentences).strip() + ('.' if not cleaned_response.endswith('.') else '')
    return f"<strong>Here is the analysis:</strong> {cleaned_response}"


def highlight_bias(text: str, bias_words: List[str]) -> str:
    # If no biased words, return the original text unaltered
    if not bias_words:
        return f"<div>{text}</div>"
    # Highlight biased words in the text
    for word in bias_words:
        text = text.replace(word, f"<span style='color: red; font-weight: bold;'>{word}</span>")
    return f"<div>{text}</div>"


def generate_response_with_model(prompt: str, progress: Callable[[float, str], None] = None) -> str:
    if FAIRSENSE_RUNTIME is None:
        initialize()
    try:
        return FAIRSENSE_RUNTIME.predict_with_text_model(prompt, progress)
    except Exception as e:
        if progress:
            progress(1.0, "Error occurred")
        return f"Error generating response: {e}"


# Governance and Safety
def ai_governance_response(prompt: str, progress: Callable[[float, str], None] = None) -> str:
    if FAIRSENSE_RUNTIME is None:
        initialize()

    response = generate_response_with_model(
        f"Provide insights and recommendations on the following AI governance and safety topic:\n\n{prompt}",
        progress=progress,
    )
    return post_process_response(response)


def analyze_text_for_bias(text_input: str, progress: gr.Progress = gr.Progress()) -> Tuple[str, str]:
    if FAIRSENSE_RUNTIME is None:
        initialize()

    progress(0, "Initializing analysis...")  # Start the progress bar

    try:
        time.sleep(0.2)  # Simulate delay for initializing
        progress(0.1, "Preparing analysis...")

        prompt = (
            f"Analyze the following text for bias. Be concise, focusing only on relevant details. Mention specific phrases or language that contribute to bias, "
            f"and describe the tone of the text. Mention who is the targeted group (if there is any group targetted) and what kind of bias it is. "
            f"Provide your response as a clear and concise paragraph. If no bias is found, state that the text appears unbiased.\n\n"
            f"Text: \"{text_input}\""
        )

        progress(0.3, "Generating response...")
        response = generate_response_with_model(
            prompt,
            progress=lambda x, desc="": progress(0.3 + x * 0.4, desc),
        )

        progress(0.7, "Post-processing response...")
        processed_response = post_process_response(response)

        progress(0.9, "Highlighting text bias...")
        bias_section = response.split("Biased words:")[-1].strip() if "Biased words:" in response else ""
        biased_words = [word.strip() for word in bias_section.split(",")] if bias_section else []
        highlighted_text = highlight_bias(text_input, biased_words)

        progress(1.0, "Analysis complete.")
        return highlighted_text, processed_response
    except Exception as e:
        progress(1.0, "Analysis failed.")
        return f"Error: {e}", ""


def preprocess_image(image: Image) -> Image:
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    # Apply median blur to reduce noise
    thresh = cv2.medianBlur(thresh, 3)
    return Image.fromarray(thresh)


def analyze_image_for_bias(image: Image, progress=gr.Progress()) -> Tuple[str, str]:
    if FAIRSENSE_RUNTIME is None:
        initialize()

    progress(0, "Initializing image analysis...")  # Start the progress bar

    try:
        time.sleep(0.1)  # Simulate delay
        progress(0.1, "Processing image...")
        image = image.convert("RGB")
        inputs = FAIRSENSE_RUNTIME.blip_processor(images=image, return_tensors="pt").to(FAIRSENSE_RUNTIME.device)

        progress(0.2, "Extracting text from image...")
        # Preprocess image for better OCR results
        preprocessed_image = preprocess_image(image)
        # OCR to extract text from the image
        extracted_text = pytesseract.image_to_string(preprocessed_image)

        progress(0.3, "Generating caption...")
        with torch.no_grad():
            caption_ids = FAIRSENSE_RUNTIME.blip_model.generate(**inputs, max_length=300, num_beams=5, temperature=0.7)
        caption_text = FAIRSENSE_RUNTIME.blip_processor.tokenizer.decode(caption_ids[0], skip_special_tokens=True).strip()

        # Combine extracted text and caption
        combined_text = f"{caption_text}. {extracted_text}"

        progress(0.6, "Analyzing combined text for bias...")
        prompt = (
            f"Analyze the following image related text for bias, mockery, disinformation, misinformation or satire. Focus only on relevant details. "
            f"Mention any specific groups that has been targetted (if any). "
            f"Provide your response as a clear and concise paragraph. If no bias is found, state that the text appears unbiased.\n\n"
            f"Text:\n\"{combined_text}\""
        )
        response = generate_response_with_model(
            prompt,
            progress=lambda x, desc="": progress(0.6 + x * 0.3, desc)
        )

        progress(0.9, "Post-processing response...")
        processed_response = post_process_response(response)

        # Extract biased words (if any)
        bias_section = response.split("Biased words:")[-1].strip() if "Biased words:" in response else ""
        biased_words = [word.strip() for word in bias_section.split(",")] if bias_section else []

        # Highlight biases in combined text
        highlighted_text = highlight_bias(combined_text, biased_words)

        progress(1.0, "Analysis complete.")
        return highlighted_text, processed_response
    except Exception as e:
        progress(1.0, f"Analysis failed: {e}")
        return f"Error: {e}", ""


# Batch Processing Functions
def analyze_text_csv(file: TextIO, output_filename: str = "analysis_results.csv") -> str:
    if FAIRSENSE_RUNTIME is None:
        initialize()

    try:
        df = pd.read_csv(file.name)
        if "text" not in df.columns:
            return "Error: The CSV file must contain a 'text' column."

        results = []
        for i, text in enumerate(df["text"]):
            try:
                highlighted_text, analysis = analyze_text_for_bias(text)
                results.append({
                    "row_index": i + 1,
                    # "original_text": text,
                    "text": highlighted_text,
                    "analysis": analysis
                })
            except Exception as e:
                results.append({
                    "row_index": i + 1,
                    # "original_text": text,
                    "text": "Error",
                    "analysis": str(e)
                })

        result_df = pd.DataFrame(results)
        # Convert DataFrame to HTML table
        html_table = result_df.to_html(escape=False)  # escape=False to render HTML in cells
        save_path = FAIRSENSE_RUNTIME.save_results_to_csv(result_df, output_filename)
        return html_table
    except Exception as e:
        return f"Error processing CSV: {e}"


def analyze_images_batch(images: List[str], output_filename: str = "image_analysis_results.csv") -> str:
    if FAIRSENSE_RUNTIME is None:
        initialize()

    try:
        results = []
        for i, image_path in enumerate(images):
            try:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                image = Image.open(image_path)
                highlighted_caption, analysis = analyze_image_for_bias(image)

                # Debugging: Log the current image being processed
                logging.info(f"Processing Image: {image_path}")

                # Convert the image to base64 for HTML display
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
        # Convert DataFrame to HTML table
        html_table = result_df.to_html(escape=False)  # escape=False to render HTML content
        save_path = FAIRSENSE_RUNTIME.save_results_to_csv(result_df, output_filename)
        return html_table
    except Exception as e:
        return f"Error processing images: {e}"


# Function to display Enhanced AI Safety Dashboard
def display_ai_safety_dashboard() -> Tuple[Figure, Figure, Figure, pd.DataFrame]:
    if FAIRSENSE_RUNTIME is None:
        initialize()

    df = pd.DataFrame(FAIRSENSE_RUNTIME.ai_safety_risks)

    # Bar Chart: Percentage Distribution of AI Risks
    fig_bar = px.bar(
        df,
        x='Risk',
        y='Percentage',
        color='Category',
        text='Percentage',
        title='Percentage Distribution of AI Safety Risks',
        labels={'Percentage': 'Percentage (%)'},
        hover_data=['Description', 'Mitigation']
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500,
        legend_title_text='Risk Category'
    )
    fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')

    # Pie Chart: Proportion of Each Risk Category
    category_counts = df['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    fig_pie = px.pie(
        category_counts,
        names='Category',
        values='Count',
        title='Proportion of Risk Categories',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Scatter Plot: Severity vs. Likelihood
    fig_scatter = px.scatter(
        df,
        x='Likelihood',
        y='Severity',
        size='Percentage',
        color='Impact',
        hover_name='Risk',
        title='Severity vs. Likelihood of AI Risks',
        labels={'Severity': 'Severity (1-10)', 'Likelihood': 'Likelihood (1-10)'},
        size_max=20
    )
    fig_scatter.update_layout(template='plotly_white', height=500)

    # Return the figures and the DataFrame
    return fig_bar, fig_pie, fig_scatter, df


# New Function: About Fairsense-AI
def display_about_page() -> str:
    about_html = """
    <style>
        .about-container {
            padding: 20px;
            font-size: 16px;
            line-height: 1.6;
        }
        .about-title {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .technology-section {
            margin-bottom: 30px;
        }
        .technology-section h3 {
            font-size: 22px;
            margin-bottom: 10px;
        }
        .technology-section p {
            margin-left: 20px;
        }
    </style>
    <div class="about-container">
        <div class="about-title">About Fairsense-AI</div>
        <div class="technology-section">
            <h3>🔍 Autoregressive Decoder-only Language Model</h3>
            <p>
                Fairsense-AI utilizes the LLMs in generating detailed analyses of textual content, detecting biases, and providing insights on AI governance topics.
            </p>
        </div>
        <div class="technology-section">
            <h3>🖼️  Image Captioning</h3>
            <p>
                Image Captioning models like Blip are used for generating descriptive captions of images. This aids in understanding the visual content and assessing it for potential biases or sensitive elements.
            </p>
        </div>
        <div class="technology-section">
            <h3>🔤 Optical Character Recognition (OCR)</h3>
            <p>
                Fairsense-AI employs OCR technology, specifically Tesseract OCR via the pytesseract library, to extract text embedded within images. This allows the tool to analyze textual content that appears within images, such as signs or documents.
            </p>
        </div>
        <div class="technology-section">
            <h3>⚙️ Transformers and PyTorch</h3>
            <p>
                The underlying models are built and run using the Transformers library by Hugging Face and PyTorch. These libraries provide robust frameworks for natural language processing and deep learning tasks.
            </p>
        </div>
        <div class="technology-section">
            <h3>📊 Plotly for Data Visualization</h3>
            <p>
                For creating interactive charts and visualizations in the AI Safety Risks Dashboard, Fairsense-AI uses Plotly, a powerful graphing library that enables interactive and informative visual representations of data.
            </p>
        </div>
        <div class="technology-section">
            <h3>💻 Gradio Interface</h3>
            <p>
                Gradio is used to build the user interface of Fairsense-AI, providing an accessible and user-friendly platform for interacting with the tool's functionalities.
            </p>
        </div>
    </div>
    """
    return about_html


def start_server(
    make_public_url: bool = True,
    allow_filesystem_access: bool = True,
    prevent_thread_lock: bool = False,
    launch_browser_on_startup: bool = False,
) -> None:
    if FAIRSENSE_RUNTIME is None:
        initialize(allow_filesystem_access=allow_filesystem_access)

    # Gradio Interface
    description = """
    <style>
        .title {
            text-align: center; 
            font-size: 3em; 
            font-weight: bold; 
            margin-bottom: 20px; 
            color: #4A90E2; /* Soft blue color */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); /* Shadow for depth */
            font-family: 'Arial', sans-serif; /* Clean, modern font */
            animation: glow 2s infinite; /* Glowing effect */
        }

        .description {
            text-align: center; 
            font-size: 1.2em; 
            margin-bottom: 40px;
            color: #333;
        }

        @keyframes glow {
            0% { text-shadow: 0 0 5px #4A90E2, 0 0 10px #4A90E2, 0 0 20px #4A90E2; }
            50% { text-shadow: 0 0 10px #4A90E2, 0 0 20px #4A90E2, 0 0 40px #4A90E2; }
            100% { text-shadow: 0 0 5px #4A90E2, 0 0 10px #4A90E2, 0 0 20px #4A90E2; }
        }
    </style>
    <div class="title">✨ Fairsense-AI ✨</div>
    <div class="description">
    Fairsense-AI is an AI-driven platform for analyzing bias in textual and visual content.  
    It is designed to promote transparency, fairness, and equity in AI systems. 
    The platform is built to align with the principles of responsible AI, with a particular focus on fairness, bias, and sustainability. </div>


    <ul>
        <li><strong>Text Analysis:</strong> Detect biases in text, highlight problematic terms, and provide actionable feedback.</li>
        <li><strong>Image Analysis:</strong> Evaluate images for embedded text and captions for bias.</li>
        <li><strong>Batch Processing:</strong> Analyze large datasets of text or images efficiently.</li>
        <li><strong>AI Governance:</strong> Gain insights into ethical AI practices and responsible deployment.</li>
    </ul>
    """

    footer = """
        <div class="footer" style="margin-top: 30px; padding-top: 10px; border-top: 1px solid #ccc;">
            <p><i>"Responsible AI adoption for a better Sustainable world."</i></p>
            <p><strong>Disclaimer:</strong> The outputs generated by this platform are based on AI models and may vary depending on the input and contextual factors. While efforts are made to ensure accuracy and fairness, users should exercise discretion and validate critical information.</p>
            <p>Contact Person: Shaina Raza, PhD, Vector Institute, email at <a href="mailto:shaina.raza@vectorinstitute.ai">shaina.raza@vectorinstitute.ai</a>.</p>

        </div>
    """

    demo = gr.Blocks(css="""
        #ai-dashboard {
            padding: 20px;
        }
        .gradio-container {
            background-color: #ffffff;
        }
    """)

    with demo:
        gr.HTML(description)
        with gr.Tabs():
            with gr.TabItem("📄 Text Analysis"):
                with gr.Row():
                    text_input = gr.Textbox(
                        lines=5,
                        placeholder="Enter text to analyze for bias",
                        label="Text Input"
                    )
                    analyze_button = gr.Button("Analyze")

                # Here we add the Examples section
                gr.Examples(
                    examples=[
                        "Some people say that women are not suitable for leadership roles.",
                        "Our hiring process is completely fair and unbiased and still we think male candidates are better based on intellect level."
                    ],
                    inputs=text_input,
                    label="Try some examples"
                )

                highlighted_text = gr.HTML(label="Highlighted Text")
                detailed_analysis = gr.HTML(label="Detailed Analysis")

                analyze_button.click(
                    analyze_text_for_bias,
                    inputs=text_input,
                    outputs=[highlighted_text, detailed_analysis],
                    show_progress=True
                )

            with gr.TabItem("🖼️ Image Analysis"):
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    analyze_image_button = gr.Button("Analyze")

                # Add example instructions with clickable download links
                gr.Markdown("""
                ### Example Images
                You can download the following images and upload them to test the analysis:
                - [Example 1](https://media.top1000funds.com/wp-content/uploads/2019/12/iStock-525807555.jpg)
                - [Example 2](https://ichef.bbci.co.uk/news/1536/cpsprodpb/BB60/production/_115786974_d6bbf591-ea18-46b9-821b-87b8f8f6006c.jpg)
                """)

                highlighted_caption = gr.HTML(label="Highlighted Text and Caption")
                image_analysis = gr.HTML(label="Detailed Analysis")

                analyze_image_button.click(
                    analyze_image_for_bias,
                    inputs=image_input,
                    outputs=[highlighted_caption, image_analysis],
                    show_progress=True
                )

            with gr.TabItem("📂 Batch Text CSV Analysis"):
                with gr.Row():
                    csv_input = gr.File(
                        label="Upload Text CSV (with 'text' column)",
                        file_types=['.csv']
                    )
                    analyze_csv_button = gr.Button("Analyze CSV")
                csv_results = gr.HTML(label="CSV Analysis Results")
                analyze_csv_button.click(
                    analyze_text_csv,
                    inputs=csv_input,
                    outputs=csv_results,
                    show_progress=True
                )
            with gr.TabItem("🗂️ Batch Image Analysis"):
                with gr.Row():
                    images_input = gr.File(
                        label="Upload Images (multiple allowed)",
                        file_types=["image"],
                        type="filepath",
                        file_count="multiple"
                    )
                    analyze_images_button = gr.Button("Analyze Images")
                images_results = gr.HTML(label="Image Batch Analysis Results")
                analyze_images_button.click(
                    analyze_images_batch,
                    inputs=images_input,
                    outputs=images_results,
                    show_progress=True
                )
            with gr.TabItem("📜 AI Governance and Safety"):
                with gr.Row():
                    # Predefined topics for the dropdown
                    predefined_topics = [
                        "Ethical AI Development",
                        "Data Privacy in AI",
                        "AI Bias Mitigation Strategies",
                        "Transparency and Explainability",
                        "Regulation and Compliance",
                        "AI in Healthcare",
                        "AI and Employment",
                        "Environmental Impact of AI",
                        "AI in Education",
                        "AI and Human Rights"
                    ]
                    # Dropdown for predefined topics
                    governance_dropdown = gr.Dropdown(
                        choices=predefined_topics,
                        label="Select a Topic",
                        value=predefined_topics[0],  # Set default value
                        interactive=True
                    )
                with gr.Row():
                    # Textbox for custom topics/questions
                    governance_input = gr.Textbox(
                        lines=3,
                        placeholder="Or enter your own topic or question about AI governance and safety...",
                        label="Custom Topic",
                        interactive=True
                    )
                governance_button = gr.Button("Get Insights")
                governance_insights = gr.HTML(label="Governance Insights")

                # Function to handle the input
                def governance_topic_handler(selected_topic: str, custom_topic: str, progress: gr.Progress = gr.Progress()):
                    progress(0, "Starting...")
                    topic = custom_topic.strip() if custom_topic.strip() else selected_topic
                    if not topic:
                        progress(1, "No topic selected")
                        return "Please select a topic from the dropdown or enter your own question."
                    progress(0.2, "Generating response...")
                    response = ai_governance_response(
                        topic,
                        progress=lambda x, desc="": progress(0.2 + x * 0.8, desc)
                    )
                    progress(1.0, "Done")
                    return response

                governance_button.click(
                    governance_topic_handler,
                    inputs=[governance_dropdown, governance_input],
                    outputs=governance_insights,
                    show_progress=True,
                )
            with gr.TabItem("📊 AI Safety Risks Dashboard"):
                fig_bar, fig_pie, fig_scatter, df = display_ai_safety_dashboard()
                gr.Markdown("### Percentage Distribution of AI Safety Risks")
                gr.Plot(fig_bar)
                # gr.Markdown("### Proportion of Risk Categories")
                # gr.Plot(fig_pie)
                gr.Markdown("### Severity vs. Likelihood of AI Risks")
                gr.Plot(fig_scatter)
                gr.Markdown("### AI Safety Risks Data")
                gr.Dataframe(df)
            with gr.TabItem("ℹ️ About Fairsense-AI"):
                about_output = gr.HTML(value=display_about_page())

        gr.HTML(footer)

    demo.queue().launch(share=make_public_url, prevent_thread_lock=prevent_thread_lock, inbrowser=launch_browser_on_startup)


if __name__ == "__main__":
    start_server()
