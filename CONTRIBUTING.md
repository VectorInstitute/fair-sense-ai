# **FairSense-AI**

## Setting up

The code is capable of running both on GPU and CPU. It will automatically detect if
there is a GPU available, and if it's not it will switch to using models that are
capable of running on CPU only.

### Installing requirements

Make a virtual environment and activate it:

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
python -m fairsenseai
```
