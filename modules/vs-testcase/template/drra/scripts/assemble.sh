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

# assemble the fabric
vesyla component assemble -a ${template_path}/arch.json -o ${workspace_path}/temp

# copy the results to the system directory
cp -r ${workspace_path}/temp/arch ${workspace_path}/system
cp -r ${workspace_path}/temp/rtl ${workspace_path}/system
cp -r ${workspace_path}/temp/isa ${workspace_path}/system
cp -r ${workspace_path}/temp/sst ${workspace_path}/system
mv ${workspace_path}/temp ${workspace_path}/archive/assemble
