#!/bin/sh
set -e

# get the script directory
script_path=$(dirname "$(realpath "$0")")
template_path=$(dirname "$script_path")
workspace_path="${template_path}/work"

# get the id of the code segment from the first argument
vsim_cli_mode="-c"

for arg in "$@"; do
  case "$arg" in
  -h | --help)
    echo "Usage: $0 <id>"
    exit 0
    ;;
  -it | -interactive | -it=all | --interactive=all | -it=rtl | --interactive=rtl)
    # set the interactive mode
    vsim_cli_mode="-voptargs=+acc -debugDB"
    ;;
  esac
done

# check the necessary directories
if [ ! -d "${workspace_path}/system/arch" ]; then
  echo "${workspace_path}/system/arch directory does not exist"
  exit 1
fi
if [ ! -d "${workspace_path}/system/firmware" ]; then
  echo "${workspace_path}/system/firmware directory does not exist"
  exit 1
fi
if [ ! -d "${workspace_path}/mem" ]; then
  echo "mem directory does not exist"
  exit 1
fi
mkdir -p ${workspace_path}/archive
mkdir -p ${workspace_path}/system/metric

# create the necessary directories
mkdir -p ${workspace_path}/temp
mkdir -p ${workspace_path}/archive/rtl_sim

# copy the necessary files
cp ${workspace_path}/system/firmware/firmware.hex temp
cp ${workspace_path}/mem/sram_image_in.bin temp
cd ${workspace_path}/temp

# gather the dependencies using bender
bender -d ${workspace_path}/system/rtl/tb script vsim -t sim >read_src.do
echo "exit" >>read_src.do

# compile the library
vsim -c -do read_src.do

# run the simulation
vsim $vsim_cli_mode -do "run -all" work.picorv_wrapper_tb

# copy the output file
cp ${workspace_path}/temp/sram_image_out.bin ${workspace_path}/mem/sram_image_m3.bin

# archive everything
mv ${workspace_path}/temp ${workspace_path}/archive/rtl_sim_${id}
