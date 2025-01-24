# FairSense AI Documentation

This folder contains the files to build the Fair Sense AI 
documentation site.

## Setting up

We are using poetry for dependency management in this project.
To install dependencies for documentation in your system,
please run:

```shell
python3 -m poetry install --with docs
```

## Running the Build

The source files use Furo, a clean and customisable Sphinx 
documentation theme.

In order to build the documentation, install the documentation 
dependencies as mentioned in the previous section, navigate to 
the docs folder and run the command:

```
make html
```