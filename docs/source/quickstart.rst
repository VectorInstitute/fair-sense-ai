Quickstart Code Examples
========================

1. **Text Bias Analysis**

   .. code-block:: python

      from fairsenseai.analysis.bias import analyze_text_for_bias

      # Example input text to analyze for bias
      text_input = "Men are naturally better at decision-making, while women excel at emotional tasks."

      # Analyze the text for bias
      highlighted_text, detailed_analysis = analyze_text_for_bias(text_input, use_summarizer=True)

      # Print the analysis results
      print("Highlighted Text:", highlighted_text)
      print("Detailed Analysis:", detailed_analysis)

2. **Image Bias Analysis**

   .. code-block:: python

      import requests
      from PIL import Image
      from io import BytesIO
      from IPython.display import display, HTML
      from fairsenseai.analysis.bias import analyze_image_for_bias

      # URL of the image to analyze.
      image_url = "https://ichef.bbci.co.uk/news/1536/cpsprodpb/BB60/production/_115786974_d6bbf591-ea18-46b9-821b-87b8f8f6006c.jpg"

      # Fetch and load the image from the URL.
      response = requests.get(image_url)
      if response.status_code == 200:
         image = Image.open(BytesIO(response.content))
         small_image = image.copy()
         small_image.thumbnail((200, 200))

         # Display the resized image and analyze for bias.
         print("Original Image (Resized):")
         display(small_image)
         highlighted_caption, image_analysis = analyze_image_for_bias(image, use_summarizer=True)

         # Print and display analysis results.
         print("\nHighlighted Caption:\n", highlighted_caption)
         print("\nImage Analysis:\n", image_analysis)
         if highlighted_caption:
            display(HTML(highlighted_caption))
         else:
            print(f"Failed to fetch the image. Status code: {response.status_code}")

3. **Insights on AI Governance & Safety**

   .. code-block:: python

      from fairsenseai.analysis.ai_governance import ai_governance_response

      # Get insights on topics related to AI governance and safety
      insights = ai_governance_response("AI Bias Mitigation Strategies")

      # Print the result
      print("AI Governance & Safety Insights:", insights)

4. **Launch the Interactive Application**

   .. code-block:: python

      from fairsenseai.app import start_server

      # Launch the Gradio application (will open in the browser)
      start_server()
