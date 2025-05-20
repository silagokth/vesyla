#!/bin/sh
set -e

# check the number of arguments
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <id>"
  exit 1
fi

# check if $0 is a number
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
  echo "Error: $1 is not a number"
  exit 1
fi

# get the id of the code segment from the first argument
id=$1
vsim_cli_mode="-c"

for arg in "$@"
do
  case "$arg" in
    -h|--help)
      echo "Usage: $0 <id>"
      exit 0
      ;;
    -it|-interactive|-it=all|--interactive=all|-it=rtl|--interactive=rtl)
      # set the interactive mode
      vsim_cli_mode=""
      ;;
  esac
done

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
echo "vsim -voptargs=+acc -debugDB work.fabric_tb" >> read_src.do
echo "log * -r" >> read_src.do
echo "run -all" >> read_src.do
vsim $vsim_cli_mode -do read_src.do
cd ..
cp temp/sram_image_out.bin mem/sram_image_m3.bin

# archive everything
mv temp archive/rtl_sim_${id}
