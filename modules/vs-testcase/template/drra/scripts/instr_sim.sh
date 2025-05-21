#!/bin/sh
set -e

# check the number of arguments
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <id>"
  exit 1
fi

# get the script directory
script_path=$(dirname "$(realpath "$0")")
template_path=$(dirname "$script_path")
workspace_path="${template_path}/work"

# get the id of the code segment from the first argument
id=$1

# check the necessary directories
if [ ! -d "${workspace_path}/system/arch" ]; then
  echo "${workspace_path}/system/arch directory does not exist"
  exit 1
fi
if [ ! -d "${workspace_path}/system/isa" ]; then
  echo "${workspace_path}/system/isa directory does not exist"
  exit 1
fi
if [ ! -d "${workspace_path}/system/instr/${id}" ]; then
  echo "${workspace_path}/system/instr/${id} directory does not exist"
  exit 1
fi
if [ ! -d "${workspace_path}/mem" ]; then
  echo "${workspace_path}/mem directory does not exist"
  exit 1
fi
if [ ! -d "${workspace_path}/system/sst" ]; then
  echo "${workspace_path}/system/sst directory does not exist"
  exit 1
fi

# create an archive directory if it does not exist
if [ ! -d "${workspace_path}/archive" ]; then
  mkdir -p ${workspace_path}/archive
fi

# create folders to store the results
mkdir -p ${workspace_path}/system/metric
mkdir -p ${workspace_path}/system/state
mkdir -p ${workspace_path}/temp
mkdir -p ${workspace_path}/archive/instr_sim_${id}

# create the output file
touch ${workspace_path}/mem/sram_image_m2.bin

# simulate the code segment
sst ${workspace_path}/system/sst/sst_sim_conf.py -- \
    --io_input_buffer_filepath ${workspace_path}/mem/sram_image_in.bin \
    --io_output_buffer_filepath ${workspace_path}/mem/sram_image_m2.bin \
    --assembly_program_path ${workspace_path}/system/instr/${id}/instr.bin

# archive everything
mv ${workspace_path}/temp ${workspace_path}/archive/instr_sim_${id}
