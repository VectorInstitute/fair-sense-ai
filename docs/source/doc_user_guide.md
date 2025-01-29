(user_guide)=

# User Guide

## pyproject.toml file and dependency management

For managing dependencies, [Poetry](https://python-poetry.org/) is the recommended tool
for our project. Install Poetry to setup the development virtual environment. Poetry
supports [optional dependency groups](https://python-poetry.org/docs/managing-dependencies/#optional-groups)
which help manage dependencies for different parts of development such as `documentation`,
`testing`, etc. The core dependencies are installed using the command:

```bash
python3 -m poetry install
```

Additional dependency groups can be installed using the `--with` flag. For example:

```bash
python3 -m poetry install --with docs
```

## documentation

The documentation source files use [Furo](https://pradyunsg.me/furo/),
a clean and customisable Sphinx documentation theme.

In order to build the documentation, install the documentation dependencies as mentioned
in the previous section, navigate to the `docs` folder and run the command:

```bash
make html
```

The documentation can be configured by updating the `docs/source/conf.py`. The markdown
files in `docs/source` can be updated to reflect the project's documentation.
