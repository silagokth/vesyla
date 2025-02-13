#!/bin/sh
set -e

mkdir -p system
mkdir -p archive

mkdir temp

vesyla-suite component assemble -a ../arch.json -o temp
cp -r temp/arch system
cp -r temp/rtl system
cp -r temp/isa system
mv temp archive/assemble
