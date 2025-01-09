# **FairSense-AI**

## Setup

The script is capable of running both on GPU and CPU. It will automatically detect if
there is a GPU available, and if it's not it will switch to using models that are
capable of running on CPU.

If you're running this in a machine with a CPU only, please follow the instructions
below. Otherwise, jump to the [Installing Requirements](#installing-requirements)
section.

### For running on CPU only

You will need to download and install Ollama
(instructions [here](https://ollama.com/download)). Make sure to also install its
CLI tool.

After that, please pre-download the Llama 3.2 model with the command below:
```shell
ollama pull llama3.2
```

### Installing requirements

First, make a virtual environment and activate it:

```shell
python -m venv venv
source venv/bin/activate
```

Install [Poetry](https://python-poetry.org/), for dependency management:

```shell
pip install --upgrade pip poetry
```

Then, install the project requirements with the command below:

```shell
poetry install --no-root
```

### Running

To run it, use the following command:

```shell

```
