#!/usr/bin/env bash

set -e

# move to the root of the project
cd "$(dirname "$0")"
cd ..

echo "Building for $(uname)"
if [ "$(uname)" == "Darwin" ]; then
  echo "Building MacOS app"

  pyinstaller installer/fairsenseai-mac.spec -y

elif [[ "$(uname)" =~ ^MINGW64_NT-10.0 ]] || [[ "$(uname)" =~ ^MSYS_NT-10.0 ]]; then
  echo "Building Windows App"
elif [ "$(uname)" == "Linux" ]; then
  echo "Building Linux App"
else
  echo "Unsupported operating system: $(uname)"
  exit 1
fi
