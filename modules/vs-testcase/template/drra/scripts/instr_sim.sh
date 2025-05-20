#!/bin/sh
set -e

# check the number of arguments
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <id>"
  exit 1
fi

# get the id of the code segment from the first argument
id=$1

# check the necessary directories
if [ ! -d "system/arch" ]; then
  echo "system/arch directory does not exist"
  exit 1
fi
if [ ! -d "system/isa" ]; then
  echo "system/isa directory does not exist"
  exit 1
fi
if [ ! -d "system/instr/${id}" ]; then
  echo "system/instr/${id} directory does not exist"
  exit 1
fi
if [ ! -d "mem" ]; then
  echo "mem directory does not exist"
  exit 1
fi
if [ ! -d "system/sst" ]; then
  echo "system/sst directory does not exist"
  exit 1
fi
mkdir -p archive
mkdir -p system/metric
mkdir -p system/state

# create the necessary directories
mkdir -p temp
mkdir -p archive/instr_sim_${id}

# simulate the code segment
touch mem/sram_image_m2.bin
sst system/sst/sst_sim_conf.py -- \
    --io_input_buffer_filepath mem/sram_image_in.bin \
    --io_output_buffer_filepath mem/sram_image_m2.bin \
    --assembly_program_path system/instr/${id}/instr.bin


# archive everything
mv temp archive/instr_sim_${id}