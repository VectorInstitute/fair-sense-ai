from typing import Optional

import gradio as gr

from fairsenseai.analysis.ai_governance import ai_governance_response
from fairsenseai.analysis.ai_safety_dashboard import display_ai_safety_dashboard
from fairsenseai.analysis.bias import (
    analyze_image_for_bias,
    analyze_images_batch,
    analyze_text_csv,
    analyze_text_for_bias,
)
from fairsenseai.runtime import get_runtime

def display_about_page() -> str:
    """
    Provides an HTML string describing the Fairsense-AI platform.

    Returns
    -------
    str
        The HTML content for the About page.
    """
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
                Fairsense-AI utilizes LLMs for generating detailed analyses of textual content,
                detecting biases, and providing insights on AI governance topics.
            </p>
        </div>
        <div class="technology-section">
            <h3>🖼️  Image Captioning</h3>
            <p>
                Fairsense-AI uses Blip for generating descriptive captions of images.
                This aids in understanding visual content and assessing it for biases or
                sensitive elements.
            </p>
        </div>
        <div class="technology-section">
            <h3>🔤 Optical Character Recognition (OCR)</h3>
            <p>
                Fairsense-AI employs Tesseract OCR to extract text from images, allowing
                analysis of textual content embedded within images.
            </p>
        </div>
        <div class="technology-section">
            <h3>⚙️ Transformers and PyTorch</h3>
            <p>
                Transformers (Hugging Face) and PyTorch power the underlying models, ensuring
                robust NLP and deep learning functionalities.
            </p>
        </div>
        <div class="technology-section">
            <h3>📊 Plotly for Data Visualization</h3>
            <p>
                Plotly is used for creating interactive charts in the AI Safety Risks Dashboard,
                providing engaging and informative data visualizations.
            </p>
        </div>
        <div class="technology-section">
            <h3>💻 Gradio Interface</h3>
            <p>
                Gradio offers a clean, user-friendly UI for interacting with the Fairsense-AI platform.
            </p>
        </div>
    </div>
    """
    return about_html


def start_server(
    make_public_url: Optional[bool] = True,
    allow_filesystem_access: Optional[bool] = True,
    prevent_thread_lock: Optional[bool] = False,
    launch_browser_on_startup: Optional[bool] = False,
) -> None:
    """
    Starts the Gradio server with multiple tabs for text analysis, image analysis,
    batch processing, AI governance insights, and an AI safety risks dashboard.

    Parameters
    ----------
    make_public_url
        Whether to make the server publicly accessible.
    allow_filesystem_access
        Whether to allow filesystem access for file uploads, required to save results
    prevent_thread_lock
        Whether to prevent thread lock issues.
    launch_browser_on_startup
        Whether to launch the browser on server startup.

    Returns
    -------
    None

    Example
    -------
    >>> start_server()
    """
    # Initialize the runtime
    get_runtime()

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
    The platform is built to align with the principles of responsible AI, with a particular focus on fairness, bias, and sustainability.
    </div>
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
<p>Developers: Shaina Raza, PhD, Vector Institute; Marelo Lotif; Mukund Sayeeganesh Chettiar.</p>
        <p>Email for Shaina Raza: <a href='mailto:shaina.raza@torontomu.ca'>shaina.raza@vectorinstitute.ai</a>.</p>
          </div>
    """

    demo = gr.Blocks(
        css="""
        #ai-dashboard {
            padding: 20px;
        }
        .gradio-container {
            background-color: #ffffff;
        }
        """
    )

    with demo:
        gr.HTML(description)

        with gr.Tabs():
            # --- Text Analysis Tab ---
            with gr.TabItem("📄 Text Analysis"):
                with gr.Row():
                    text_input = gr.Textbox(
                        lines=5,
                        placeholder="Enter text to analyze for bias",
                        label="Text Input"
                    )
                    # Summarizer toggle for text analysis
                    use_summarizer_checkbox_text = gr.Checkbox(
                        value=True,
                        label="Use Summarizer?"
                    )
                    analyze_button = gr.Button("Analyze")

                # Examples
                gr.Examples(
                    examples=[
                        "Some people say that women are not suitable for leadership roles.",
                        "Our hiring process is fair and unbiased, but we prefer male candidates for their intellect level."
                    ],
                    inputs=text_input,
                    label="Try some examples"
                )

                highlighted_text = gr.HTML(label="Highlighted Text")
                detailed_analysis = gr.HTML(label="Detailed Analysis")

                analyze_button.click(
                    analyze_text_for_bias,
                    inputs=[text_input, use_summarizer_checkbox_text],
                    outputs=[highlighted_text, detailed_analysis],
                    show_progress=True
                )

            # --- Image Analysis Tab ---
            with gr.TabItem("🖼️ Image Analysis"):
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    # Summarizer toggle for image analysis
                    use_summarizer_checkbox_img = gr.Checkbox(
                        value=True,
                        label="Use Summarizer?"
                    )
                    analyze_image_button = gr.Button("Analyze")

                # Example images
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
                    inputs=[image_input, use_summarizer_checkbox_img],
                    outputs=[highlighted_caption, image_analysis],
                    show_progress=True
                )

            # --- Batch Text CSV Analysis Tab ---
            with gr.TabItem("📂 Batch Text CSV Analysis"):
                with gr.Row():
                    csv_input = gr.File(
                        label="Upload Text CSV (with 'text' column)",
                        file_types=['.csv']
                    )
                    # Summarizer toggle for batch text CSV
                    use_summarizer_checkbox_text_csv = gr.Checkbox(
                        value=True,
                        label="Use Summarizer?"
                    )
                    analyze_csv_button = gr.Button("Analyze CSV")

                csv_results = gr.HTML(label="CSV Analysis Results")

                analyze_csv_button.click(
                    analyze_text_csv,
                    inputs=[csv_input, use_summarizer_checkbox_text_csv],
                    outputs=csv_results,
                    show_progress=True
                )

            # --- Batch Image Analysis Tab ---
            with gr.TabItem("🗂️ Batch Image Analysis"):
                with gr.Row():
                    images_input = gr.File(
                        label="Upload Images (multiple allowed)",
                        file_types=["image"],
                        type="filepath",
                        file_count="multiple"
                    )
                    # Summarizer toggle for batch image
                    use_summarizer_checkbox_img_batch = gr.Checkbox(
                        value=True,
                        label="Use Summarizer?"
                    )
                    analyze_images_button = gr.Button("Analyze Images")

                images_results = gr.HTML(label="Image Batch Analysis Results")

                analyze_images_button.click(
                    analyze_images_batch,
                    inputs=[images_input, use_summarizer_checkbox_img_batch],
                    outputs=images_results,
                    show_progress=True
                )

            # --- AI Governance and Safety ---
            with gr.TabItem("📜 AI Governance and Safety"):
                with gr.Row():
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
                    governance_dropdown = gr.Dropdown(
                        choices=predefined_topics,
                        label="Select a Topic",
                        value=predefined_topics[0],
                        interactive=True
                    )
                with gr.Row():
                    governance_input = gr.Textbox(
                        lines=3,
                        placeholder="Or enter your own topic or question about AI governance and safety...",
                        label="Custom Topic",
                        interactive=True
                    )
                # Summarizer toggle for AI Governance
                use_summarizer_checkbox_governance = gr.Checkbox(
                    value=True,
                    label="Use Summarizer?"
                )
                governance_button = gr.Button("Get Insights")
                governance_insights = gr.HTML(label="Governance Insights")

                def governance_topic_handler(
                    selected_topic: str,
                    custom_topic: str,
                    use_summarizer: bool,
                    progress: gr.Progress = gr.Progress()
                ):
                    progress(0, "Starting...")
                    topic = custom_topic.strip() if custom_topic.strip() else selected_topic
                    if not topic:
                        progress(1, "No topic selected")
                        return "Please select a topic from the dropdown or enter your own question."

                    progress(0.2, "Generating response...")
                    # Pass the summarizer toggle
                    response = ai_governance_response(
                        topic,
                        use_summarizer=use_summarizer,
                        progress=lambda x, desc="": progress(0.2 + x * 0.8, desc)
                    )
                    progress(1.0, "Done")
                    return response

                governance_button.click(
                    governance_topic_handler,
                    inputs=[governance_dropdown, governance_input, use_summarizer_checkbox_governance],
                    outputs=governance_insights,
                    show_progress=True
                )

            # --- AI Safety Risks Dashboard ---
            with gr.TabItem("📊 AI Safety Risks Dashboard"):
                fig_bar, fig_pie, fig_scatter, df = display_ai_safety_dashboard()
                gr.Markdown("### Percentage Distribution of AI Safety Risks")
                gr.Plot(fig_bar)
                # If you'd like to show the pie chart, you can uncomment:
                # gr.Markdown("### Proportion of Risk Categories")
                # gr.Plot(fig_pie)
                gr.Markdown("### Severity vs. Likelihood of AI Risks")
                gr.Plot(fig_scatter)
                gr.Markdown("### AI Safety Risks Data")
                gr.Dataframe(df)

            # --- About Page ---
            with gr.TabItem("ℹ️ About Fairsense-AI"):
                about_output = gr.HTML(value=display_about_page())

        gr.HTML(footer)

    demo.queue().launch(share=make_public_url, prevent_thread_lock=prevent_thread_lock, inbrowser=launch_browser_on_startup)


if __name__ == "__main__":
    start_server()