#!/usr/bin/env bash

# This script is used to create an AppImage of vesyla

set -e
set -x

# Get paths
BUILDDIR=$(pwd)/build
APPDIR=$BUILDDIR/appdir
CARGODIR=$BUILDDIR/cargo
SCRIPTDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"

# Create directories
mkdir -p ${BUILDDIR}
if [ -d $APPDIR ]; then
  rm -rf $APPDIR
fi
mkdir -p $APPDIR
mkdir -p $CARGODIR

cd $BUILDDIR

# Install rust
if ! command -v rustc &>/dev/null; then
  echo "Installing rust..."
  CARGO_HOME=$CARGODIR RUSTUP_HOME=$CARGODIR curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source $HOME/.cargo/env
fi

# Install python library and requirements
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install --force-reinstall -r $SCRIPTDIR/requirements.txt

# Compile the application to a fakeroot directory: $APPDIR
cmake -DCMAKE_INSTALL_PREFIX=/usr ..
make -j$(nproc)
make DESTDIR=$APPDIR install

# Download the linuxdeploy tool
wget -c https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-x86_64.AppImage

# Create the AppImage and rename it to vesyla
./linuxdeploy-x86_64.AppImage --appimage-extract-and-run --appdir $APPDIR --output appimage
mv vesyla-x86_64.AppImage ../vesyla
