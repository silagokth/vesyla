#!/usr/bin/env bash

set -e

# Install dependencies for the project depending on the OS

if [ -f /etc/lsb-release ]; then
    # Ubuntu
    sudo apt-get update
    sudo apt-get install -y make gcc g++ cmake libboost-all-dev protobuf-compiler libprotobuf-dev python3 python3-pip
elif [ -f /etc/debian_version ]; then
    # Debian
    sudo apt-get update
    sudo apt-get install -y make gcc g++ cmake libboost-all-dev protobuf-compiler libprotobuf-dev python3 python3-pip
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


# Install rust
if ! command -v rustc &> /dev/null; then
    echo "Installing rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Get directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# install python library according to python_requirements.txt
pip3 install --upgrade pip
pip3 install -U -r $DIR/python_requirements.txt