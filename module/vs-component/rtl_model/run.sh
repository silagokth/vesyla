#!/usr/bin/env bash

# exit on error
set -e

# compile program
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
vesyla-suite schedule -p pasm/0.pasm -c pasm/0.cstr -o compile
vesyla-suite manas -i compile/0.asm -s isa.json -o compile
cp compile/instr.bin ./instr.bin

while read line; do
    # if starts with #, skip
    if [[ $line == \#* ]]; then
        continue
    fi
    vlog -sv ${line}
    if [ $? -ne 0 ]; then
        echo "Error in compiling ${line}"
        return 1
    fi
done < hierarchy.txt

vlog -sv ../output/rtl/fabric_tb.sv

vsim -voptargs=+acc -debugDB -do "log * -r;run -all" work.fabric_tb



