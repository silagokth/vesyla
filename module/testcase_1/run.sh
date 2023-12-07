#!/usr/bin/bash

set -e

# compile the code
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python3 ../vs-manas/src/main.py -i org_instr.txt -s isa.json -o .
python3 ../vs-alimpsim/src/main.py --arch arch.json --instr instr.bin --isa isa.json --input input.txt --output output.txt --metric metric.json