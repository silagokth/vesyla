#!/usr/bin/env bash

# This script is used to create an AppImage of the application.

set -e

BUILDDIR=$(pwd)/build
APPDIR=$BUILDDIR/appdir
mkdir -p ${BUILDDIR}
if [ -d $APPDIR ]; then
  rm -rf $APPDIR
fi
mkdir -p $APPDIR

# compile the application to a fakeroot directory: $APPDIR
cd $BUILDDIR
cmake -DCMAKE_INSTALL_PREFIX=/usr ..
make -j$(nproc)
make DESTDIR=$APPDIR install

# download the linuxdeploy tool
wget -c https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-x86_64.AppImage

# create the AppImage and rename it to vesyla
./linuxdeploy-x86_64.AppImage --appimage-extract-and-run --appdir $APPDIR --output appimage
mv vesyla-x86_64.AppImage ../vesyla