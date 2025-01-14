import torch
import ollama
from PIL import Image
import gradio as gr
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Configuration
MODEL_ID = 'llama3.2-vision'  # Unified model for text and image analysis
client = ollama.Client()

# Unified Text Analysis Function
def analyze_text_for_bias(text_input, progress=gr.Progress()):
    progress(0, "Initializing text analysis...")
    try:
        prompt = (
            f"Analyze the following text for bias. Identify specific phrases, the tone, and any targeted groups. "
            f"Provide a concise analysis. If unbiased, state that explicitly.\n\nText: \"{text_input}\""
        )
        response = client.chat(model=MODEL_ID, messages=[{"role": "user", "content": prompt}])
        result = response.get("message", {}).get("content", "No response from the model.")
        progress(1.0, "Text analysis complete.")
        # Format the result as HTML
        return f"<div style='font-family: Arial; font-size: 14px;'><b>Analysis Result:</b><br>{result}</div>"
    except Exception as e:
        progress(1.0, "Text analysis failed.")
        logging.error(f"Error during text analysis: {e}")
        return f"<div style='color: red;'>Error: {e}</div>"

# Unified Image Analysis Function
def analyze_image_for_bias(image, progress=gr.Progress()):
    progress(0, "Initializing image analysis...")
    try:
        # Save the image temporarily for the API call
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # Construct the API request
        prompt = (
            "Analyze the content of the uploaded image. Assess the image for potential indications of bias, "
            "disinformation, or satire. Consider the visual elements, symbols, and composition. Provide a "
            "detailed yet concise analysis, highlighting any possible implications or messages conveyed."
        )

        # Call the model API
        response = client.chat(
            model=MODEL_ID,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [temp_image_path]  # Pass the image path
            }]
        )

        # Extract and return the response content
        result = response.get("message", {}).get("content", "No response from the model.")
        progress(1.0, "Image analysis complete.")
        # Format the result as HTML
        return f"<div style='font-family: Arial; font-size: 14px;'><b>Analysis Result:</b><br>{result}</div>"
    except Exception as e:
        progress(1.0, "Image analysis failed.")
        logging.error(f"Error during image analysis: {e}")
        return f"<div style='color: red;'>Error: {e}</div>"
    finally:
        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# Gradio Interface with Resizing During Upload
def resize_image_on_upload(image):
    # Resize the image during upload for consistent processing
    max_size = (350, 350)
    return image.resize(max_size, Image.LANCZOS)

def main():
    description = """
    <h1>Fairsense-AI</h1>
    <p>An AI platform for detecting bias in text and images using Llama 3.2 Vision.</p>
    """
    demo = gr.Blocks()

    with demo:
        gr.HTML(description)

        with gr.Tabs():
            with gr.TabItem("Text Analysis"):
                text_input = gr.Textbox(lines=5, placeholder="Enter text for bias analysis")
                text_output = gr.HTML(label="Analysis Result")  # Updated to use HTML for styled output
                gr.Button("Analyze").click(
                    analyze_text_for_bias,
                    inputs=text_input,
                    outputs=text_output,
                    show_progress=True,
                )

            with gr.TabItem("Image Analysis"):
                # Resized image display in Gradio
                image_input = gr.Image(type="pil", label="Upload Image", width=300, height=300)
                image_output = gr.HTML(label="Analysis Result")  # Updated to use HTML for styled output
                gr.Button("Analyze").click(
                    lambda image: analyze_image_for_bias(resize_image_on_upload(image)),
                    inputs=image_input,
                    outputs=image_output,
                    show_progress=True,
                )

    demo.queue().launch(share=True)

if __name__ == "__main__":
    main()
