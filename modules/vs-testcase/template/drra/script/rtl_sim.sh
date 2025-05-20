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
mkdir -p archive
mkdir -p system/metric
mkdir -p system/state

# create the necessary directories
mkdir -p temp
mkdir -p archive/rtl_sim_${id}

# simulate the code segment
cp system/instr/${id}/instr.bin temp
cp mem/sram_image_in.bin temp
cd temp
bender -d ../system/rtl/tb script vsim -t sim > read_src.do
vsim -c -voptargs=+acc -debugDB -do read_src.do -do "log * -r;run -all" work.fabric_tb
cd ..
cp temp/sram_image_out.bin mem/sram_image_m3.bin

# archive everything
mv temp archive/rtl_sim_${id}
