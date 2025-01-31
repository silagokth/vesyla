#!/bin/sh
set -e

# check the number of arguments
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_dir>"
  exit 1
fi

# get the input directory
input_dir=$1

# check the necessary directories
if [ ! -d "system/arch" ]; then
  echo "system/arch directory does not exist"
  exit 1
fi
if [ ! -d "system/isa" ]; then
  echo "system/isa directory does not exist"
  exit 1
fi
mkdir -p archive

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

  # create the necessary directories
  mkdir -p temp

  # schedule, assemble, and simulate the code segment
  vesyla-suite schedule \
      -a system/arch/arch.json \
      -p ../pasm/${id}.pasm \
      -c ../pasm/${id}.cstr \
      -o temp
  vesyla-suite manas \
      -a system/arch/arch.json \
      -i system/isa/isa.json \
      -s temp/0.asm \
      -o temp

  # preserve the instructions
  mkdir -p system/instr/${id}
  cp temp/instr.bin system/instr
  cp temp/instr.txt system/instr

  # archive everything
  mv temp archive/compile_${id}

done
