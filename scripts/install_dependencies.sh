#!/usr/bin/env bash

set -e
set -x

# Install dependencies for the project depending on the OS

if [ -f /etc/lsb-release ]; then
  # Ubuntu
  sudo apt-get update
  sudo apt-get install -y cmake curl python3 python3-pip python3-venv protobuf-compiler libprotobuf-dev wget libfuse2 file
  #sudo apt-get install -y make gcc g++ cmake libboost-all-dev protobuf-compiler libprotobuf-dev python3 python3-pip libfuse2
elif [ -f /etc/debian_version ]; then
  # Debian
  sudo apt-get update
  sudo apt-get install -y make gcc g++ cmake libboost-all-dev protobuf-compiler libprotobuf-dev python3 python3-pip libfuse2
elif [ -f /etc/fedora-release ]; then
  # Fedora
  sudo dnf install -y python python3-pip make gcc gcc-c++ cmake boost boost-devel protobuf protobuf-devel
elif [ -f /etc/arch-release ]; then
  # Arch
  sudo pacman -Syu --noconfirm make gcc cmake boost protobuf python python-pip
else
  echo "Unsupported distribution! You need to install the dependencies manually. Check the requirements.txt."
  exit 1
fi
