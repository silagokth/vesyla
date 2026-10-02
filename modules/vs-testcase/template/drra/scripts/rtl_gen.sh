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

# the sized architecture the compiler wrote for this program
arch_sized="${workspace_path}/system/instr/${id}/arch_sized.json"
if [ ! -f "${arch_sized}" ]; then
  echo "${arch_sized} does not exist"
  exit 1
fi

# create the necessary directories
mkdir -p ${workspace_path}/archive
mkdir -p ${workspace_path}/temp

# generate the RTL for the fabric sized for this program
vesyla fabric rtl -a ${arch_sized} -o ${workspace_path}/temp

# copy the results next to the program
cp -r ${workspace_path}/temp/rtl ${workspace_path}/system/instr/${id}
mv ${workspace_path}/temp ${workspace_path}/archive/rtl_gen_${id}
