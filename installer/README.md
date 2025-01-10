# FairSense AI Installers

This folder contains the files needed to build the Fair Sense AI
installers for multiple platforms.

The installer will be built for the same system installed in
the machine being used to run the build script.

## Setting up

We are using [PyInstaller](https://pyinstaller.org) as the build
tool for the installer. To install `pyinstaller` in your system,
please run:

```shell
pip install -U pyinstaller
```

## Running the Build

To trigger a build of the installer, please run the following
command from the root folder:

```shell
pyinstaller installer/fairsenseai.spec -y
```
