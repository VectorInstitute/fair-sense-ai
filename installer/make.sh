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
  echo "Logs can be located at installer\make.log"
  
  pyinstaller installer/fairsenseai-win.spec -y > installer/make.log 2>&1

elif [ "$(uname)" == "Linux" ]; then
  echo "Building Linux App"

  pyinstaller installer/fairsenseai-linux.spec -y

else
  echo "Unsupported operating system: $(uname)"
  exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
  echo "Compressing MacOS app"
  cd ./dist
  tar czpvf FairSenseAI.tgz FairSenseAI
  cd ..
fi
