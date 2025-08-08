#!/usr/bin/env bash

set -e
set -x

# Install dependencies for the project depending on the OS

if [ -f /etc/lsb-release ]; then
  # Ubuntu
  sudo apt-get update
  sudo apt-get install -y \
    clang cmake \
    curl file wget \
    libfuse2 flex bison
elif [ -f /etc/fedora-release ]; then
  # Fedora
  sudo dnf install -y make gcc gcc-c++ cmake boost boost-devel flex bison
elif [ -f /etc/arch-release ]; then
  # Arch
  sudo pacman -Syu --noconfirm make gcc cmake boost flex bison
else
  echo "Unsupported distribution! You need to install the dependencies manually."
  exit 1
fi
