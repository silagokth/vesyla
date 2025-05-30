#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

spin_animation() {
  tput civis
  (
    trap '' EXIT
    trap 'exit' TERM
    spinner=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    while true; do
      for i in "${spinner[@]}"; do
        printf "\\r${BLUE}%s${NC} " "$i"
        sleep 0.1
      done
    done
  ) &
  SPIN_PID=$! # Store the PID of the spinner process
}

start_spinner() {
  if [ $debug_mode = false ]; then
    spin_animation
  fi
}

stop_spinner() {
  if [ $debug_mode = true ]; then
    printf "\n"
    return 0 # If debug mode is enabled, do not stop the spinner
  fi
  if [ -n "$SPIN_PID" ] && kill -0 "$SPIN_PID" 2>/dev/null; then
    kill "$SPIN_PID" 2>/dev/null
    wait "$SPIN_PID" 2>/dev/null
  fi
  tput cnorm # Show cursor
  # If the first arg is empty print a checkmark
  if [ "$1" -eq 0 ]; then
    printf "\\r${GREEN}✓${NC} \n"
  else
    printf "\\r${RED}✗${NC} \n"
  fi
}

cleanup() {
  exit_code=$1
  stop_spinner $exit_code
  exit $exit_code
}

trap 'cleanup $1' INT TERM

# Variables
interactive_mode=false
debug_mode=false

# Get the script full path
template_path=$(dirname "$(realpath "$0")")

# Argument parsing
for arg in "$@"; do
  case "$arg" in
  -nc | --no-color | --nocolor)
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
    echo "INFO: No color mode enabled"
    ;;
  esac
done
for arg in "$@"; do
  case "$arg" in
  -it | --interactive | -it=all | --interactive=all)
    interactive_mode=all
    printf "${CYAN}INFO:${NC} Interactive mode enabled for SST and RTL\n"
    ;;
  -it=sst | --interactive=sst)
    interactive_mode=sst
    printf "${CYAN}INFO:${NC} Interactive mode enabled for SST\n"
    ;;
  -it=rtl | --interactive=rtl)
    interactive_mode=rtl
    printf "${CYAN}INFO:${NC} Interactive mode enabled for RTL\n"
    ;;
  -d | --debug)
    debug_mode=true
    printf "${CYAN}INFO:${NC} Debug mode enabled\n"
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
    stop_spinner $status
    printf " ${RED}-> ERROR:${NC} $desc failed!\n"
    echo "$output"
    exit $status
  fi
}

# Prepare environment
rm -rf ${template_path}/work
mkdir -p ${template_path}/work
cd ${template_path}/work
mkdir -p mem

# Assemble the fabric
start_spinner
printf "  ${BOLD}Assembling the fabric${NC}"
if [ "$debug_mode" = true ]; then
  stop_spinner 0
  bash ${template_path}/scripts/assemble.sh
  start_spinner
else
  run_and_check "Assembly" bash ${template_path}/scripts/assemble.sh
fi
stop_spinner 0

# Model 0
printf "${BOLD}Model 0:${NC} C++ implementation\n"
## Compile
start_spinner
printf "  ${BLUE}Compiling${NC}"
run_and_check "Compilation" g++ -g -I${template_path}/model_0/include -o run_model_0 ${template_path}/model_0/main.cpp ${template_path}/model_0/src/Drra.cpp ${template_path}/model_0/src/Util.cpp
stop_spinner 0

## Run
start_spinner
printf "  ${BLUE}Running${NC}"
./run_model_0
stop_spinner 0

## Check that the output exists
start_spinner
printf "  ${BLUE}Verifying${NC} (mem/sram_image_m0.bin)"
if [ ! -f "mem/sram_image_m0.bin" ]; then
  printf " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin not found!"
  exit 1
fi
## Check that the output is not empty
if [ ! -s "mem/sram_image_m0.bin" ]; then
  printf " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin is empty!"
  exit 1
fi
sort -n mem/sram_image_m0.bin -o mem/sram_image_m0.bin
stop_spinner 0

# Model 1
printf "${BOLD}Model 1:${NC} ${YELLOW}Warning${NC} Not implemented. Skipping...\n"
cp mem/sram_image_m0.bin mem/sram_image_m1.bin

# Model 2
printf "${BOLD}Model 2:${NC} instruction-level simulation\n"
## Compile

start_spinner
printf "  ${BLUE}Compiling${NC}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/compile.sh ${template_path}/pasm
else
  run_and_check "Compilation" bash ${template_path}/scripts/compile.sh ${template_path}/pasm
fi
stop_spinner 0

## Run
start_spinner
if [ "$interactive_mode" = "all" ] || [ "$interactive_mode" = "sst" ]; then
  printf "  ${YELLOW}Warning:${NC} SST interactive mode is not implemented yet.\n  Running in non-interactive mode...\n"
fi
printf "  ${BLUE}Running${NC}"
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/instr_sim.sh 0
else
  run_and_check "Instruction-level simulation" bash ${template_path}/scripts/instr_sim.sh 0
fi
stop_spinner 0

## Verify (compare to model 0 output)
start_spinner
printf "  ${BLUE}Verifying${NC} (mem/sram_image_m2.bin)"
sort -n mem/sram_image_m2.bin -o mem/sram_image_m2.bin
set +e
error_output=$(diff -q mem/sram_image_m0.bin mem/sram_image_m2.bin 2>&1)
if [ $? -ne 0 ]; then
  printf " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin and mem/sram_image_m2.bin differ!"
  printf "${RED}Error details:${NC}"
  echo "$error_output"
  exit 1
fi
set -e
stop_spinner 0

# Model 3
printf "${BOLD}Model 3:${NC} RTL simulation\n"
## Run
start_spinner
if [ "$interactive_mode" = "all" ] || [ "$interactive_mode" = "rtl" ]; then
  printf "  ${BLUE}Compiling & Running${NC} (interactive mode)"
else
  printf "  ${BLUE}Compiling & Running${NC}"
fi
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/rtl_sim.sh 0 -it="$interactive_mode"
else
  run_and_check "RTL simulation" bash ${template_path}/scripts/rtl_sim.sh 0 -it="$interactive_mode"
fi
stop_spinner 0

## Verify (compare to model 0 output)
printf "  ${BLUE}Verifying${NC} (mem/sram_image_m3.bin)"
sed -i 's/^[ \t]*//' mem/sram_image_m3.bin             # Remove leading whitespace from the output file
sort -n mem/sram_image_m3.bin -o mem/sram_image_m3.bin # Reorder the memory file
set +e
error_output=$(diff -q mem/sram_image_m0.bin mem/sram_image_m3.bin 2>&1)
if [ $? -ne 0 ]; then
  printf " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin and mem/sram_image_m3.bin differ!"
  printf "${RED}Error details:${NC}"
  echo "$error_output"
  exit 1
fi
set -e
stop_spinner 0

printf "\n${GREEN}All models executed successfully!${NC}\n\n"

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
