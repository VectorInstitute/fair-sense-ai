from typing import Optional, Tuple
from pathlib import Path
from functools import partial

import gradio as gr

from fairsenseai.analysis.ai_safety_dashboard import display_ai_safety_dashboard
from fairsenseai.analysis.bias import (
    analyze_image_for_bias,
    analyze_images_batch,
    analyze_text_csv,
    analyze_text_for_bias,
)
from fairsenseai.analysis.risk_assessment import analyze_text_for_risks
from fairsenseai.runtime import get_runtime

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
    run_time = get_runtime(allow_filesystem_access=allow_filesystem_access)

    script_dir = Path(__file__).resolve().parent
    ui_dir = script_dir / "ui"

    with open(ui_dir / "home.html") as home_file:
        home = home_file.read()

    with open(ui_dir / "footer.html") as footer_file:
        footer = footer_file.read()

    with open(ui_dir / "about.html") as page_file:
        about = page_file.read()

    demo = gr.Blocks()
    with demo:
        gr.HTML(home)
        gr.HTML(footer)

    with demo.route("Bias Identification", "/bias"):
        gr.Markdown("# Bias Identification", elem_classes="page-title")

        with gr.Tabs():
            # --- Text Analysis Tab ---
            with gr.TabItem("📄 Text Analysis"):
                with gr.Row():
                    text_input = gr.Textbox(
                        lines=5,
                        placeholder="Enter text to analyze for bias",
                        label="Text Input",
                    )
                    # Summarizer toggle for text analysis
                    use_summarizer_checkbox_text = gr.Checkbox(
                        value=True,
                        label="Use Summarizer?",
                    )
                    analyze_button = gr.Button("Analyze")

                # Examples
                gr.Examples(
                    examples=[
                        "Some people say that women are not suitable for leadership roles.",
                        "Our hiring process is fair and unbiased, but we prefer male candidates "
                        "for their intellect level.",
                    ],
                    inputs=text_input,
                    label="Try some examples",
                )

                highlighted_text = gr.HTML(label="Highlighted Text")
                detailed_analysis = gr.HTML(label="Detailed Analysis")
                score = gr.Number(label="Bias Score (%) [0 = Unbiased, 100 = Highly Biased]", precision=0)

                analyze_button.click(
                    analyze_text_for_bias,
                    inputs=[text_input, use_summarizer_checkbox_text],
                    outputs=[highlighted_text, detailed_analysis, score],
                    show_progress=True,
                )

            # --- Image Analysis Tab ---
            with gr.TabItem("🖼️ Image Analysis"):
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    # Summarizer toggle for image analysis
                    use_summarizer_checkbox_img = gr.Checkbox(
                        value=True,
                        label="Use Summarizer?",
                    )
                    analyze_image_button = gr.Button("Analyze")

                # Example images
                gr.Markdown(
                """
                ### Example Images
                You can download the following images and upload them to test the analysis:
                - [Example 1](https://media.top1000funds.com/wp-content/uploads/2019/12/iStock-525807555.jpg)
                - [Example 2](https://ichef.bbci.co.uk/news/1536/cpsprodpb/BB60/production/_115786974_d6bbf591-ea18-46b9-821b-87b8f8f6006c.jpg)
                """
                )

                highlighted_caption = gr.HTML(label="Highlighted Text and Caption")
                image_analysis = gr.HTML(label="Detailed Analysis")

                analyze_image_button.click(
                    analyze_image_for_bias,
                    inputs=[image_input, use_summarizer_checkbox_img],
                    outputs=[highlighted_caption, image_analysis],
                    show_progress=True,
                )

            # --- Batch Text CSV Analysis Tab ---
            with gr.TabItem("📂 Batch Text CSV Analysis"):
                with gr.Row():
                    csv_input = gr.File(
                        label="Upload Text CSV (with 'text' column)",
                        file_types=[".csv"],
                    )
                    # Summarizer toggle for batch text CSV
                    use_summarizer_checkbox_text_csv = gr.Checkbox(
                        value=True,
                        label="Use Summarizer?",
                    )
                    analyze_csv_button = gr.Button("Analyze CSV")

                csv_results = gr.HTML(label="CSV Analysis Results")
                
                gr.Markdown("**Note:** A bias score of 0% means the text is unbiased, while 100% indicates high bias.")

                analyze_csv_button.click(
                    analyze_text_csv,
                    inputs=[csv_input, use_summarizer_checkbox_text_csv],
                    outputs=csv_results,
                    show_progress=True,
                )

            # --- Batch Image Analysis Tab ---
            with gr.TabItem("🗂️ Batch Image Analysis"):
                with gr.Row():
                    images_input = gr.File(
                        label="Upload Images (multiple allowed)",
                        file_types=["image"],
                        type="filepath",
                        file_count="multiple",
                    )
                    # Summarizer toggle for batch image
                    use_summarizer_checkbox_img_batch = gr.Checkbox(
                        value=True,
                        label="Use Summarizer?",
                    )
                    analyze_images_button = gr.Button("Analyze Images")

                images_results = gr.HTML(label="Image Batch Analysis Results")

                analyze_images_button.click(
                    analyze_images_batch,
                    inputs=[images_input, use_summarizer_checkbox_img_batch],
                    outputs=images_results,
                    show_progress=True,
                )

        gr.HTML(footer)

    with demo.route("Risk Management", "/risk"):
        gr.Markdown("# Risk Management", elem_classes="page-title")

        with gr.Tabs():
            with gr.TabItem("📄 Risk Identification and Mitigation"):
                with gr.Row():
                    text_input = gr.Textbox(
                        lines=5,
                        placeholder="Enter text to analyze for bias",
                        label="Text Input",
                        scale=4,
                    )
                    analyze_button = gr.Button("Analyze Risks", scale=1)

                # Examples
                gr.Examples(
                    examples=[
                        "Our team is creating a healthcare chatbot that analyzes patient symptoms using electronic "
                        "health records. It provides early diagnoses, treatment advice, and can access sensitive "
                        "patient information. We must handle data security, privacy, and potential misdiagnoses.",
                        "We’re building an AI-powered facial recognition tool to improve workplace security. "
                        "It will monitor employee entrances, verify identities in real-time, and store face embeddings."
                        " The system must comply with privacy regulations and handle sensitive biometrics.",
                    ],
                    inputs=text_input,
                    label="Try some examples",
                )

                csv_folder_path = None
                if allow_filesystem_access:
                    csv_output_file = gr.File(
                        label="Risks and Outcomes Traceability Matrix"
                    )
                    csv_folder_path = run_time.risk_default_directory
                else:
                    print("Not saving results to CSV because filesystem access is not allowed.")

                highlighted_text = gr.HTML(label="Highlighted Text")

                outputs = [highlighted_text, csv_output_file] if allow_filesystem_access else [highlighted_text]

                # Partial function with csv_folder_path already provided
                analyze_func = partial(analyze_text_for_risks, csv_folder_path=csv_folder_path)

                analyze_button.click(
                    analyze_func,
                    inputs=[text_input],
                    outputs=outputs,
                    show_progress=True,
                )

                gr.Markdown(
                    """
                    **Useful References:**
                    - [MIT AI Risk Repository](https://airisk.mit.edu/)
                    - [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
                    """
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

        gr.HTML(footer)

    with demo.route("About FairSense-AI", "/about"):
        gr.Markdown("# About FairSense-AI", elem_classes="page-title")
        gr.HTML(value=about)
        gr.HTML(footer)

    demo.queue().launch(
        share=make_public_url,
        prevent_thread_lock=prevent_thread_lock,
        inbrowser=launch_browser_on_startup,
        allowed_paths=run_time.get_allowed_paths(),
    )


if __name__ == "__main__":
    start_server()
