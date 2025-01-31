#!/bin/sh
set -e
rm -rf work
mkdir -p work
cd work
mkdir -p mem
g++ -g -Wall -Wno-strict-aliasing -Wno-unknown-pragmas -I../include -O2 -o main ../main.cpp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
./main
