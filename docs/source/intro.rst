FairSense-AI
============

FairSense-AI is a cutting-edge, AI-driven platform designed to promote transparency, fairness, and equity by analyzing bias in textual and visual content. Built with sustainability in mind, it leverages energy-efficient AI frameworks to ensure an eco-friendly approach to tackling societal biases.

---

Installation and Setup
======================

Step 1: Install Supporting Tools
--------------------------------

1. **Python 3.9+**

   Ensure Python is installed. Download it from `Python's official website <https://www.python.org/downloads/>`_.

2. **Tesseract OCR**

   Required for extracting text from images.

   **Installation Instructions**:

   - **Ubuntu**:

     .. code-block:: bash

        sudo apt-get update
        sudo apt-get install tesseract-ocr

   - **macOS (Homebrew)**:

     .. code-block:: bash

        brew install tesseract

   - **Windows**:  
     Download and install Tesseract OCR from `this link <https://github.com/UB-Mannheim/tesseract/wiki>`_.

3. **Ollama (for CPU only)**

   Ollama is a tool that easily installs versions of Llama that are capable of running on CPU. If the machine does not have a GPU, this is a required step.

   - Download and install Ollama from `Ollama's official website <https://ollama.com/download>`_. Make sure to also install the CLI tool.

   - Pre-download the Llama 3.2 model with the command below:

     .. code-block:: shell

        ollama pull llama3.2

4. **Optional (GPU Acceleration)**

   Install PyTorch with CUDA support:

   .. code-block:: bash

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

Step 2: Install FairSense-AI
----------------------------

Install the ``fair-sense-ai`` package using pip:

.. code-block:: bash

   pip install fair-sense-ai

---

Quickstart Code Examples
========================

1. **Text Bias Analysis**

   .. code-block:: python

      from fairsenseai import analyze_text_for_bias

      # Example input text to analyze for bias
      text_input = "Men are naturally better at decision-making, while women excel at emotional tasks."

      # Analyze the text for bias
      highlighted_text, detailed_analysis = analyze_text_for_bias(text_input)

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
      highlighted_caption, image_analysis = analyze_image_for_bias(image)

      # Print the analysis results
      print("Highlighted Caption:", highlighted_caption)
      print("Image Analysis:", image_analysis)

3. **Launch the Interactive Application**

   .. code-block:: python

      from fairsenseai import start_server

      # Launch the Gradio application (will open in the browser)
      start_server()

---

Bias Detection Tutorial
=======================

**Data and Sample Notebooks**

1. **Download the Data**:  
   `Google Drive Link <https://drive.google.com/drive/folders/1_D7lTz-TC6yhV7xsZIDzk-tJvl4TAwyi?usp=sharing>`_

2. **Colab Notebook**:  
   `Run the Tutorial <https://colab.research.google.com/drive/1en8JtZTAIa5MuV5OZWYNteYl95Ql9xy7?usp=sharing>`_

---

Usage Instructions
==================

**Launching the Application**

Run the following command to start Fair-Sense-AI:

.. code-block:: bash

   fair-sense-AI

This will launch the Gradio-powered interface in your default web browser.

---

Features
========

1. **Text Analysis**

   - Input or paste text in the **Text Analysis** tab.
   - Click **Analyze** to detect and highlight biases.

2. **Image Analysis**

   - Upload an image in the **Image Analysis** tab.
   - Click **Analyze** to detect biases in embedded text or captions.

3. **Batch Text CSV Analysis**

   - Upload a CSV file with a ``text`` column in the **Batch Text CSV Analysis** tab.
   - Click **Analyze CSV** to process all entries.

4. **Batch Image Analysis**

   - Upload multiple images in the **Batch Image Analysis** tab.
   - Click **Analyze Images** for a detailed review.

5. **AI Governance Insights**

   - Navigate to the **AI Governance and Safety** tab.
   - Choose a predefined topic or input your own query.
   - Click **Get Insights** for recommendations.

---

Additional Setup in Colab
=========================

Run the following commands to ensure everything is ready:

.. code-block:: bash

   !pip install --quiet fair-sense-ai
   !pip uninstall sympy -y
   !pip install sympy --upgrade
   !apt update
   !apt install -y tesseract-ocr

**Note**: Restart your system if you're using Google Colab.

---

Troubleshooting
===============

- **Slow Model Download**:  
  Ensure a stable internet connection for downloading models.

- **Tesseract OCR Errors**:  
  Verify Tesseract is installed and accessible in your system's PATH.

- **GPU Support**:  
  Use the CUDA-compatible version of PyTorch for better performance.

---

Contact
=======

For inquiries or support, contact:  
**Shaina Raza, PhD**  
Applied ML Scientist, Responsible AI  
`shaina.raza@vectorinstitute.ai <mailto:shaina.raza@torontomu.ca>`_

---

License
=======

This project is licensed under the `Creative Commons License <https://creativecommons.org/licenses/>`_.

📚 `Documentation <https://vectorinstitute.github.io/FairSense-AI/>`__
======================================================================

.. |PyPI| image:: https://img.shields.io/pypi/v/cycquery
   :target: https://pypi.org/project/fair-sense-ai
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/cycquery
.. .. |code checks| image:: https://github.com/VectorInstitute/cycquery/actions/workflows/code_checks.yml/badge.svg
..    :target: https://github.com/VectorInstitute/cycquery/actions/workflows/code_checks.yml
.. .. |integration tests| image:: https://github.com/VectorInstitute/cycquery/actions/workflows/integration_tests.yml/badge.svg
..    :target: https://github.com/VectorInstitute/cycquery/actions/workflows/integration_tests.yml
.. .. |docs| image:: https://github.com/VectorInstitute/cycquery/actions/workflows/docs_deploy.yml/badge.svg
..    :target: https://github.com/VectorInstitute/cycquery/actions/workflows/docs_deploy.yml
.. .. |codecov| image:: https://codecov.io/gh/VectorInstitute/cycquery/branch/main/graph/badge.svg
..    :target: https://codecov.io/gh/VectorInstitute/cycquery
.. .. |license| image:: https://img.shields.io/github/license/VectorInstitute/cycquery.svg
..    :target: https://github.com/VectorInstitute/cycquery/blob/main/LICENSE