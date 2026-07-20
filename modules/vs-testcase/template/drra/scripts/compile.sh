#!/bin/sh
set -e

# check the number of arguments
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <input_dir> [-d|--debug]"
  exit 1
fi

# get the script directory
script_path=$(dirname "$(realpath "$0")")
template_path=$(dirname "$script_path")
workspace_path="${template_path}/work"

# get the input directory
input_dir=$1
shift

# parse remaining flags
debug_flag=""
for arg in "$@"; do
  case "$arg" in
  -d | --debug)
    debug_flag="-d"
    ;;
  esac
done

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
pasm_files=$(ls ${input_dir}/*.pasm ${input_dir}/*.mlir 2>/dev/null || true)

# get the name of each file without extension, also remove the path
ids=$(echo ${pasm_files} | tr ' ' '\n' | sed 's/\.\(pasm\|mlir\)//g' | sed 's/.*\///g' | sort -u)

# check if id in ids is number or not
for id in ${ids}; do
  if ! [[ ${id} =~ ^[0-9]+$ ]]; then
    echo "${id} is not a number. PASM and CSTR files must be named with numbers."
    exit 1
  fi
done

# for each id, compile
for id in ${ids}; do
  # create a temp directory for each id
  mkdir -p ${workspace_path}/temp

  # schedule, assemble the code segment
  if [ -f "${template_path}/pasm/${id}.mlir" ]; then
    vesyla compile \
      -a ${workspace_path}/system/arch/arch.json \
      -i ${workspace_path}/system/isa/isa.json \
      -m ${template_path}/pasm/${id}.mlir \
      -o ${workspace_path}/temp \
      ${debug_flag}
  else
    vesyla compile \
      -a ${workspace_path}/system/arch/arch.json \
      -i ${workspace_path}/system/isa/isa.json \
      -p ${template_path}/pasm/${id}.pasm \
      -o ${workspace_path}/temp \
      ${debug_flag}
  fi

  # preserve the instructions
  mkdir -p ${workspace_path}/system/instr/${id}
  cp ${workspace_path}/temp/instr.bin system/instr/${id}
  cp ${workspace_path}/temp/instr.asm system/instr/${id}

  # archive everything
  mv ${workspace_path}/temp ${workspace_path}/archive/compile_${id}
done
