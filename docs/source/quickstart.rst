Quickstart Code Examples
========================

1. **Text Bias Analysis**

   .. code-block:: python

      from fairsenseai import analyze_text_for_bias

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
      from fairsenseai import analyze_image_for_bias

      # URL of the image to analyze
      image_url = "https://media.top1000funds.com/wp-content/uploads/2019/12/iStock-525807555.jpg"

      # Fetch and load the image
      response = requests.get(image_url)
      image = Image.open(BytesIO(response.content))

      # Analyze the image for bias
      highlighted_caption, image_analysis = analyze_image_for_bias(image, use_summarizer=True)

      # Print the analysis results
      print("Highlighted Caption:", highlighted_caption)
      print("Image Analysis:", image_analysis)

3. **Insights on AI Governance & Safety**

   .. code-block:: python

      from fairsenseai import ai_governance_response

      # Get insights on topics related to AI governance and safety
      insights = ai_governance_response("AI Bias Mitigation Strategies")

      # Print the result
      print("AI Governance & Safety Insights:", insights)

4. **Launch the Interactive Application**

   .. code-block:: python

      from fairsenseai import start_server

      # Launch the Gradio application (will open in the browser)
      start_server()