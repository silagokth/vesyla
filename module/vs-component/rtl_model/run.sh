#!/usr/bin/env bash

# exit on error
set -e

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

vlog -sv fabric_tb.sv

vsim -c -voptargs=+acc -debugDB -do "log * -r;run 400ns" work.fabric_tb



