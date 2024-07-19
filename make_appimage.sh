#!/usr/bin/env bash

# This script is used to create an AppImage of the application.

# compile the application to a fakeroot directory: appdir
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr ..
make -j$(nproc)
make DESTDIR=appdir install

# download the linuxdeploy tool
wget -c https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-x86_64.AppImage

# create the AppImage and rename it to vesyla-suite
./linuxdeploy-x86_64.AppImage --appdir appdir --output appimage
mv vesyla-suite-x86_64.AppImage vesyla-suite