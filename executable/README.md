# FairSense AI Executables

This folder contains the files needed to build the Fair Sense AI
executables for multiple platforms.

The executable will be built for the same system installed in
the machine being used to run the build script.

## Setting up

We are using [PyInstaller](https://pyinstaller.org) as the build
tool for the executable. To install `pyinstaller` in your system,
please run:

```shell
pip install -U pyinstaller
```

## Running the Build

To trigger a build of the executable, please run the following
command from the root folder:

```shell
./executable/make.sh
```
