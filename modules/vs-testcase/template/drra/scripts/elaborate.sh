#!/bin/sh
set -e

# get path to the script
script_path=$(dirname "$(realpath "$0")")
template_path=$(dirname "$script_path")
workspace_path="${template_path}/work"

# create the necessary directories
mkdir -p ${workspace_path}/system
mkdir -p ${workspace_path}/archive
mkdir -p ${workspace_path}/temp

# elaborate the fabric: the arch.json and ISA the compiler reads. The SST and
# RTL are generated per program after it is compiled (sst_gen.sh, rtl_gen.sh).
vesyla fabric elaborate -a ${template_path}/arch.json -o ${workspace_path}/temp

# copy the results to the system directory
cp -r ${workspace_path}/temp/arch ${workspace_path}/system
cp -r ${workspace_path}/temp/isa ${workspace_path}/system
mv ${workspace_path}/temp ${workspace_path}/archive/elaborate
