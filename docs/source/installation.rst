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