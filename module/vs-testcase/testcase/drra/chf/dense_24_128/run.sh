#!/bin/sh
set -e
rm -rf work
mkdir work
cd work
cp ../main.cpp ./
cp ../include/*.hpp ./
g++ -g -Wall -Wno-strict-aliasing -Wno-unknown-pragmas -O2 -o main main.cpp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
./main
