#!/bin/sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Variables
interactive_mode=false
debug_mode=false

# Get the script full path
template_path=$(dirname $(realpath "$0"))

# Argument parsing
for arg in "$@"; do
  case "$arg" in
  -it | --interactive | -it=all | --interactive=all)
    interactive_mode=all
    echo "${CYAN}INFO:${NC} Interactive mode enabled for SST and RTL"
    ;;
  -it=sst | --interactive=sst)
    interactive_mode=sst
    echo "${CYAN}INFO:${NC} Interactive mode enabled for SST"
    ;;
  -it=rtl | --interactive=rtl)
    interactive_mode=rtl
    echo "${CYAN}INFO:${NC} Interactive mode enabled for RTL"
    ;;
  -d | --debug)
    debug_mode=true
    echo "${CYAN}INFO:${NC} Debug mode enabled"
    ;;
  -nc | --no-color)
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
    echo "INFO: No color mode enabled"
    ;;
  -h | --help)
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -it, --interactive=all|sst|rtl   Enable interactive mode for SST or RTL or both (all); default is off"
    echo "  -d, --debug                      Enable debug mode; default is off"
    echo "  -nc, --no-color                  Disable colored output; default is on"
    echo "  -h, --help                       Show this help message and exit"
    exit 0
    ;;
  esac
done

# Function to run commands and check for errors
run_and_check() {
  # Usage: run_and_check "description" command [args...]
  desc="$1"
  shift
  set +e
  output=$("$@" 2>&1)
  status=$?
  set -e
  if [ $status -ne 0 ]; then
    echo " ${RED}-> ERROR:${NC} $desc failed!"
    echo "${RED}Error details:${NC}"
    echo "$output"
    exit 1
  fi
}

# Prepare environment
rm -rf ${template_path}/work
mkdir -p ${template_path}/work
cd ${template_path}/work
mkdir -p mem

# Assemble the fabric
echo -n "${BOLD}Assembling the fabric${NC}"
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/assemble.sh
else
  run_and_check "Assembly" bash ${template_path}/scripts/assemble.sh
fi
echo " ${GREEN}-> OK${NC}"

# Model 0
echo "${BOLD}Model 0:${NC} C++ implementation"
## Compile
echo -n "\t${BLUE}Compiling${NC}"
run_and_check "Compilation" g++ -g -I${template_path}/model_0/include -o run_model_0 ${template_path}/model_0/main.cpp ${template_path}/model_0/src/Drra.cpp ${template_path}/model_0/src/Util.cpp
echo " ${GREEN}-> OK${NC}"

## Run
echo -n "\t${BLUE}Running${NC}"
./run_model_0
echo " ${GREEN}-> OK${NC}"

## Check that the output exists
echo -n "\t${BLUE}Verifying${NC} (mem/sram_image_m0.bin)"
if [ ! -f "mem/sram_image_m0.bin" ]; then
  echo " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin not found!"
  exit 1
fi
## Check that the output is not empty
if [ ! -s "mem/sram_image_m0.bin" ]; then
  echo " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin is empty!"
  exit 1
fi
sort -n mem/sram_image_m0.bin -o mem/sram_image_m0.bin
echo " ${GREEN}-> OK${NC}"

# Model 1
echo "${YELLOW}Warning:${NC} Model 1 is not implemented yet. Skipping..."
cp mem/sram_image_m0.bin mem/sram_image_m1.bin

# Model 2
echo "${BOLD}Model 2:${NC} instruction-level simulation"
## Compile
echo -n "\t${BLUE}Compiling${NC}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/compile.sh ${template_path}/pasm
else
  run_and_check "Compilation" bash ${template_path}/scripts/compile.sh ${template_path}/pasm
fi
echo " ${GREEN}-> OK${NC}"

## Run
if [ "$interactive_mode" = "all" ] || [ "$interactive_mode" = "sst" ]; then
  echo "\t${YELLOW}Warning:${NC} SST interactive mode is not implemented yet.\n\tRunning in non-interactive mode..."
fi
echo -n "\t${BLUE}Running${NC}"
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/instr_sim.sh 0
else
  run_and_check "Instruction-level simulation" bash ${template_path}/scripts/instr_sim.sh 0
fi
echo " ${GREEN}-> OK${NC}"

## Verify (compare to model 0 output)
echo -n "\t${BLUE}Verifying${NC} (mem/sram_image_m2.bin)"
sort -n mem/sram_image_m2.bin -o mem/sram_image_m2.bin
set +e
error_output=$(diff -q mem/sram_image_m0.bin mem/sram_image_m2.bin 2>&1)
if [ $? -ne 0 ]; then
  echo " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin and mem/sram_image_m2.bin differ!"
  echo "${RED}Error details:${NC}"
  echo "$error_output"
  exit 1
fi
set -e
echo " ${GREEN}-> OK${NC}"

# Model 3
echo "${BOLD}Model 3:${NC} RTL simulation"
## Run
if [ "$interactive_mode" = "all" ] || [ "$interactive_mode" = "rtl" ]; then
  echo -n "\t${BLUE}Compiling & Running${NC} (interactive mode)"
else
  echo -n "\t${BLUE}Compiling & Running${NC}"
fi
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/rtl_sim.sh 0 -it="$interactive_mode"
else
  run_and_check "RTL simulation" bash ${template_path}/scripts/rtl_sim.sh 0 -it="$interactive_mode"
fi
echo " ${GREEN}-> OK${NC}"

## Verify (compare to model 0 output)
echo -n "\t${BLUE}Verifying${NC} (mem/sram_image_m3.bin)"
sed -i 's/^[ \t]*//' mem/sram_image_m3.bin             # Remove leading whitespace from the output file
sort -n mem/sram_image_m3.bin -o mem/sram_image_m3.bin # Reorder the memory file
set +e
error_output=$(diff -q mem/sram_image_m0.bin mem/sram_image_m3.bin 2>&1)
if [ $? -ne 0 ]; then
  echo " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin and mem/sram_image_m3.bin differ!"
  echo "${RED}Error details:${NC}"
  echo "$error_output"
  exit 1
fi
set -e
echo " ${GREEN}-> OK${NC}"

echo "\n${GREEN}All models executed successfully!${NC}\n"

cat <<"EOF"
                        ░█████████▒░
                      ░███████████████▓
                    ░▓███████████████████░
                  ▓███████████████████████▓
                  ▒█████████████████████████▓
                  ▓█████▓            ░███████░
                  ▓████▒             ▒█████▓
                    ████▒             ░█████
                    ███░░▒░         ▒▓▓██▓
                    ░▒▒█░▒░▓███  ▒███▓░▒▓█░▒
                    ░ ░░ ░░▒░   ▓░▒▒▓░ ░ ▒░
                      ▓        ░ ▓░░░   ░ ░
                      ░░░     ░░  ▓▒    ▒░░
                        ▒  ▒████▒█████▒░▓
                        ▒░▒████████████▒▓
                        ▒░  ░▓ ░ ▒█░░▒▓░
                        ░▒    ▒▓▓▓  ░▒█
      ▒▒▓░               ▓▒░░      ▓▒██▒                  ▓▒▓░
      ░▒  ▒              ▓█▒ ░▓▒▓▓███▒░▓██▒               ▒  ░▒
      ▒  ▒░          ░█▓█████░      ▓█████░▒            ░▒  ▒
    ░░▒▓  ▒       ░▒█  ▓████████▓▓████████▒░░▒▒░        ▒  ▓▒░░
    ▒░     ▒▒  ▒▒▒░▒    ░██████░   ████████▒     ░▒▒▒▒  ▒▒     ░▒
  ▒  ░ ░░▒▒▒     ▒      ▒██████▓ ▓████████▒       ░    ▒▒▒░     ▒░
  ▒░ ░▒░░░ ▒▒    ▒       ░███████░█████████░░░     ░░   ▒▒ ░░░▒░  ▒
  ▒       ▓▒▒    ▒▒░      ▓█████▓ ▒███████▓ ▒     ▒▒░   ▒░▓     ░░▒
  ▒     ░▓▒▒       ▒░     ░█████   ▒██████▒ ▒    ░░      ▒▒█░     ▒
  ░█▒░▒▓▒▒▒░     ░░   ░░   █████    ▓█████░        ▒░     ▒▒▒▓▓░▒█░
  ███▓▓▒▒█░    ░▒     ░   ░███▓    ░█████      ░░  ▒    ░█▓▒▒▓███░
  ░██▒░░ ▓▒▒ ▓░  ▒░    ▒░   ▓██▒     ▓███░         ▒░ ░▓ ░░▒ ░░▒██░
   _____                _      _____                             _
  / ____|              | |    / ____|                           | |
 | |  __ _ __ ___  __ _| |_  | (___  _   _  ___ ___ ___  ___ ___| |
 | | |_ | '__/ _ \/ _` | __|  \___ \| | | |/ __/ __/ _ \/ __/ __| |
 | |__| | | |  __/ (_| | |_   ____) | |_| | (_| (_|  __/\__ \__ \_|
  \_____|_|  \___|\__,_|\__| |_____/ \__,_|\___\___\___||___/___(_)
EOF
