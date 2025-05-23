#!/bin/sh
set -e

# check the number of arguments
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_dir>"
  exit 1
fi

# get the script directory
script_path=$(dirname "$(realpath "$0")")
template_path=$(dirname "$script_path")
workspace_path="${template_path}/work"

# get the input directory
input_dir=$1

# check the necessary directories
if [ ! -d "${workspace_path}/system/arch" ]; then
  echo "${workspace_path}/system/arch directory does not exist"
  exit 1
fi
if [ ! -d "${workspace_path}/system/isa" ]; then
  echo "${workspace_path}/system/isa directory does not exist"
  exit 1
fi

# create an archive directory if it does not exist
if [ ! -d "${workspace_path}/archive" ]; then
  mkdir -p ${workspace_path}/archive
fi

# get all .pasm files as list
pasm_files=$(ls ${input_dir}/*.pasm)

# get the name of each file without extension, also remove the path
ids=$(echo ${pasm_files} | sed 's/\.pasm//g' | sed 's/.*\///g')

# check if id in ids is number or not
for id in ${ids}; do
  if ! [[ ${id} =~ ^[0-9]+$ ]]; then
    echo "${id} is not a number. PASM and CSTR files must be named with numbers."
    exit 1
  fi
done

# check .cstr file exists for each .pasm file
for id in ${ids}; do
  if [ ! -f ${input_dir}/${id}.cstr ]; then
    echo "${input_dir}/${id}.cstr does not exist"
    exit 1
  fi
done

# for each id, compile
for id in ${ids}; do
  # create a temp directory for each id
  mkdir -p ${workspace_path}/temp

  # schedule, assemble, and simulate the code segment
  vesyla schedule \
    -a ${workspace_path}/system/arch/arch.json \
    -p ${template_path}/pasm/${id}.pasm \
    -c ${template_path}/pasm/${id}.cstr \
    -o ${workspace_path}/temp
  vesyla manas \
    -a ${workspace_path}/system/arch/arch.json \
    -i ${workspace_path}/system/isa/isa.json \
    -s ${workspace_path}/temp/0.asm \
    -o ${workspace_path}/temp

  # preserve the instructions
  mkdir -p ${workspace_path}/system/instr/${id}
  cp ${workspace_path}/temp/instr.bin system/instr/${id}
  cp ${workspace_path}/temp/instr.txt system/instr/${id}

  # archive everything
  mv ${workspace_path}/temp ${workspace_path}/archive/compile_${id}
done
