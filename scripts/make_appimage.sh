#!/usr/bin/env bash

# This script is used to create an AppImage of vesyla

set -e
# set -x

# Get paths
SCRIPTDIR="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
BUILDDIR=$SCRIPTDIR/../build
APPDIR=$BUILDDIR/appdir
LOGFILE=${BUILDDIR}/build.log
if [ -f $LOGFILE ]; then
  rm -f $LOGFILE
fi

# Create directories
mkdir -p ${BUILDDIR}
if [ -d $APPDIR ]; then
  rm -rf $APPDIR
fi
mkdir -p $APPDIR

touch $LOGFILE

# Direct all stdout and stderr to the log file
exec > >(tee -a $LOGFILE) 2>&1

# Function to print log messages.
# It takes two arguments: the log level and the message.
# It prints the message to the console and appends it to the log file.
log() {
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  BLUE='\033[0;34m'
  NC='\033[0m'
  # check if it has two arguments
  if [ "$#" -ne 2 ]; then
    echo "${RED}[fatal]${NC}: Invalid number of arguments. Usage: log <level> <message>"
    exit 1
  fi
  local level=$1
  local message=$2
  if [ "$level" = "info" ]; then
    echo -e "${GREEN}[$level]${NC}: $message"
  elif [ "$level" = "warn" ]; then
    echo -e "${YELLOW}[$level]${NC}: $message"
  elif [ "$level" = "error" ]; then
    echo -e "${RED}[$level]${NC}: $message"
  elif [ "$level" = "fatal" ]; then
    echo -e "${RED}[$level]${NC}: $message"
  else
    echo "${RED}[fatal]${NC}: Invalid log level: $level"
    exit 1
  fi
  if [ "$level" = "fatal" ]; then
    echo "${RED}[fatal]${NC}: Fatal error detected, exiting..."
    echo "${RED}[fatal]${NC}: Check the ${LOGFILE} to find out the detailed error!"
    exit 1
  fi
}

cd $BUILDDIR

# Install rust
if ! command -v rustc &>/dev/null; then
  log info "Rust is not installed. Installing rust..."
  echo "Installing rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source $HOME/.cargo/env
else
  log info "Rust is already installed. Skipping installation."
fi

# Compile the application to a fakeroot directory: $APPDIR
log info "Compiling the application..."
cmake -DCMAKE_INSTALL_PREFIX=/usr ..
cmake --build . -- -j$(nproc)
make DESTDIR=$APPDIR install

# Download the linuxdeploy tool
# Check if linuxdeploy tool is already downloaded
if [ -f "linuxdeploy-x86_64.AppImage" ] && [ -x "linuxdeploy-x86_64.AppImage" ]; then
  log info "linuxdeploy tool already exists. Skipping download."
else
  log info "Downloading linuxdeploy tool..."
  wget -c https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
fi
chmod +x linuxdeploy-x86_64.AppImage

# Create the AppImage and rename it to vesyla
log info "Creating the AppImage..."
./linuxdeploy-x86_64.AppImage --appimage-extract-and-run --appdir $APPDIR --output appimage
mv vesyla-x86_64.AppImage ../vesyla

# Show success message
echo "AppImage created successfully!"
../vesyla --version
