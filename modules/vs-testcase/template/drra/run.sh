#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires bash to run. Please use bash to execute it."
  exit 1
fi

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
  if [ "$debug_mode" = false ]; then
    spin_animation
  fi
}

stop_spinner() {
  if [ "$debug_mode" = true ]; then
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

trap 'cleanup $?' INT TERM

# Variables
interactive_mode=false
debug_mode=false
models="0,2,3"

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
  -m=* | --models=*)
    models="${arg#*=}"
    ;;
  -d | --debug)
    debug_mode=true
    printf "${CYAN}INFO:${NC} Debug mode enabled\n"
    ;;
  -h | --help)
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -it, --interactive=all|sst|rtl   Enable interactive mode for SST or RTL or both (all); default is off"
    echo "  -m, --models=LIST                Models to run, comma-separated from 0,2,3; default 0,2,3."
    echo "                                   Model 0 is required: it writes the input memory image"
    echo "                                   (mem/sram_image_in.bin) and the reference output."
    echo "  -d, --debug                      Enable debug mode; default is off"
    echo "  -nc, --no-color                  Disable colored output; default is on"
    echo "  -h, --help                       Show this help message and exit"
    exit 0
    ;;
  esac
done

# Models to run (--models). Model 0 always runs: its C++ model writes the input image
# the simulators read and the output image they are checked against.
run_m2=false
run_m3=false
IFS=',' read -ra model_list <<<"$models"
has_m0=false
for m in "${model_list[@]}"; do
  case "$m" in
  0) has_m0=true ;;
  2) run_m2=true ;;
  3) run_m3=true ;;
  *)
    printf "${RED}ERROR:${NC} unknown model '%s' in --models=%s (choose from 0,2,3)\n" "$m" "$models"
    exit 1
    ;;
  esac
done
if [ "$has_m0" = false ]; then
  printf "${RED}ERROR:${NC} --models=%s: model 0 is required (it writes the input and reference images)\n" "$models"
  exit 1
fi
if [ "$models" != "0,2,3" ]; then
  printf "${CYAN}INFO:${NC} Models: %s\n" "$models"
fi

# Enable DRRA SST monitoring (per-cycle prints + JSON trace file) only in debug
# mode. The instruction-level (SST) model reads VESYLA_DEBUG; leaving it unset
# keeps the default fast path (monitoring off). Exported so child scripts
# (instr_sim.sh -> sst) inherit it.
if [ "$debug_mode" = true ]; then
  export VESYLA_DEBUG=1
fi

# Function to run commands and check for errors
run_and_check() {
  # Usage: run_and_check "description" [fail_code] command [args...]
  # If the second argument is numeric, exit with it on failure; otherwise default to 1 (setup).
  desc="$1"
  shift
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    fail_code="$1"
    shift
  else
    fail_code=1
  fi
  set +e
  output=$("$@" 2>&1)
  status=$?
  set -e
  if [ $status -ne 0 ]; then
    stop_spinner $status
    printf " ${RED}-> ERROR:${NC} $desc failed!\n"
    echo "$output"
    exit "$fail_code"
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
  stop_spinner 1
  printf " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin not found!"
  exit 1
fi
## Check that the output is not empty
if [ ! -s "mem/sram_image_m0.bin" ]; then
  stop_spinner 1
  printf " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin is empty!"
  exit 1
fi
sort -n mem/sram_image_m0.bin -o mem/sram_image_m0.bin
python3 ${template_path}/scripts/dump_sram_image.py mem/sram_image_m0.bin --data-type int16_t || true
stop_spinner 0

# Model 1
printf "${BOLD}Model 1:${NC} ${YELLOW}Warning${NC} Not implemented. Skipping...\n"
cp mem/sram_image_m0.bin mem/sram_image_m1.bin
python3 ${template_path}/scripts/dump_sram_image.py mem/sram_image_m1.bin --data-type int16_t || true

# Model 2
if [ "$run_m2" = true ]; then
printf "${BOLD}Model 2:${NC} instruction-level simulation\n"
## Compile

start_spinner
printf "  ${BLUE}Compiling${NC}"
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/compile.sh ${template_path}/pasm -d || exit 2
else
  run_and_check "Compilation" 2 bash ${template_path}/scripts/compile.sh ${template_path}/pasm
fi
stop_spinner 0

## Run
start_spinner
if [ "$interactive_mode" = "all" ] || [ "$interactive_mode" = "sst" ]; then
  printf "  ${YELLOW}Warning:${NC} SST interactive mode is not implemented yet.\n  Running in non-interactive mode...\n"
fi
printf "  ${BLUE}Running${NC}"
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/instr_sim.sh 0 || exit 2
else
  run_and_check "Instruction-level simulation" 2 bash ${template_path}/scripts/instr_sim.sh 0
fi
stop_spinner 0

## Verify (compare to model 0 output)
start_spinner
printf "  ${BLUE}Verifying${NC} (mem/sram_image_m2.bin)"
sort -n mem/sram_image_m2.bin -o mem/sram_image_m2.bin
python3 ${template_path}/scripts/dump_sram_image.py mem/sram_image_m2.bin --data-type int16_t || true
set +e
error_output=$(diff -q mem/sram_image_m0.bin mem/sram_image_m2.bin 2>&1)
if [ $? -ne 0 ]; then
  stop_spinner 1
  printf " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin and mem/sram_image_m2.bin differ!\n"
  printf "${RED}Error details:${NC}"
  echo "$error_output"
  exit 3
fi
set -e
stop_spinner 0
else
  printf "${BOLD}Model 2:${NC} skipped (--models=%s)\n" "$models"
fi

# Model 3
if [ "$run_m3" = true ]; then
printf "${BOLD}Model 3:${NC} RTL simulation\n"
## Compile (done by model 2 when it runs)
if [ "$run_m2" = false ]; then
start_spinner
printf "  ${BLUE}Compiling${NC}"
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/compile.sh ${template_path}/pasm -d || exit 2
else
  run_and_check "Compilation" 2 bash ${template_path}/scripts/compile.sh ${template_path}/pasm
fi
stop_spinner 0
fi
## Run
start_spinner
if [ "$interactive_mode" = "all" ] || [ "$interactive_mode" = "rtl" ]; then
  printf "  ${BLUE}Compiling & Running${NC} (interactive mode)"
else
  printf "  ${BLUE}Compiling & Running${NC}"
fi
if [ "$debug_mode" = true ]; then
  bash ${template_path}/scripts/rtl_sim.sh 0 -d -it="$interactive_mode" || exit 4
else
  run_and_check "RTL simulation" 4 bash ${template_path}/scripts/rtl_sim.sh 0 -it="$interactive_mode"
fi
stop_spinner 0

## Verify (compare to model 0 output)
printf "  ${BLUE}Verifying${NC} (mem/sram_image_m3.bin)"
sed -i 's/^[ \t]*//' mem/sram_image_m3.bin             # Remove leading whitespace from the output file
sort -n mem/sram_image_m3.bin -o mem/sram_image_m3.bin # Reorder the memory file
set +e
error_output=$(diff -q mem/sram_image_m0.bin mem/sram_image_m3.bin 2>&1)
if [ $? -ne 0 ]; then
  stop_spinner 1
  printf " ${RED}-> ERROR:${NC} mem/sram_image_m0.bin and mem/sram_image_m3.bin differ!\n"
  printf "${RED}Error details:${NC}"
  echo "$error_output"
  exit 5
fi
set -e
stop_spinner 0
else
  printf "${BOLD}Model 3:${NC} skipped (--models=%s)\n" "$models"
fi

# Timing validation: the instruction-level (SST) and RTL models must agree on
# the realized cycle count. Both simulators write it to a file every run: the
# SST controller writes `instr_sim_cycles.txt` (_currentSSTCycle/10) at
# teardown, the RTL testbench writes `rtl_sim_cycles.txt`. A mismatch means the
# compiler schedule, the SST model and the RTL hardware disagree on timing.
if [ "$run_m2" = true ] && [ "$run_m3" = true ]; then
printf "${BOLD}Timing:${NC} SST (instruction-level) vs RTL cycle count\n"
start_spinner
printf "  ${BLUE}Verifying${NC} (SST == RTL)"
sst_cycles=""
rtl_cycles=""
if [ -f "instr_sim_cycles.txt" ]; then
  sst_cycles=$(tr -d '[:space:]' <instr_sim_cycles.txt)
fi
rtl_cycles_file=$(find archive -name 'rtl_sim_cycles.txt' 2>/dev/null | head -1)
if [ -n "$rtl_cycles_file" ] && [ -f "$rtl_cycles_file" ]; then
  rtl_cycles=$(tr -d '[:space:]' <"$rtl_cycles_file")
fi
if [ -z "$sst_cycles" ] || [ -z "$rtl_cycles" ]; then
  stop_spinner 1
  printf " ${RED}-> ERROR:${NC} cycle count missing (SST='%s' RTL='%s')\n" "${sst_cycles:-N/A}" "${rtl_cycles:-N/A}"
  exit 1
fi
if [ "$sst_cycles" -ne "$rtl_cycles" ]; then
  stop_spinner 1
  printf " ${RED}-> ERROR:${NC} cycle mismatch: SST=%s RTL=%s (diff=%s)\n" "$sst_cycles" "$rtl_cycles" "$((rtl_cycles - sst_cycles))"
  exit 1
fi
stop_spinner 0
printf "  ${CYAN}%s${NC} cycles (SST == RTL)\n" "$sst_cycles"
else
  printf "${BOLD}Timing:${NC} SST vs RTL check needs models 2 and 3; skipped\n"
  for f in instr_sim_cycles.txt $(find archive -name 'rtl_sim_cycles.txt' 2>/dev/null | head -1); do
    if [ -f "$f" ]; then printf "  ${CYAN}%s${NC} cycles (%s)\n" "$(tr -d '[:space:]' <"$f")" "$f"; fi
  done
fi

# Cycle-accurate trace comparison (debug mode only). The RTL VCD (all resource
# interface signals + cycle_count) and the SST Chrome-trace are both produced
# only under -d, so this step is gated on debug_mode. It projects both onto a
# canonical per-cycle event schema and reports where SST and RTL disagree,
# tolerating a single constant skew (see scripts/trace_compare.py). It is a
# report by default: the memory result and total cycle count are already gated
# above; per-cycle micro-divergences are surfaced here for inspection, not
# treated as run failures (use trace_compare.py --strict to gate on them).
if [ "$debug_mode" = true ] && [ "$run_m2" = true ] && [ "$run_m3" = true ]; then
  printf "${BOLD}Trace:${NC} cycle-accurate SST vs RTL comparison\n"
  vcd_file=$(find archive -name 'trace.vcd' 2>/dev/null | head -1)
  sst_trace="trace_complete.json"
  if [ -f "$sst_trace" ] && [ -n "$vcd_file" ] && [ -f "$vcd_file" ]; then
    python3 ${template_path}/scripts/trace_extract_sst.py "$sst_trace" \
      -o trace_sst_canonical.json
    python3 ${template_path}/scripts/trace_extract_rtl.py "$vcd_file" \
      -o trace_rtl_canonical.json
    printf "  ${BLUE}Comparing${NC}\n  "
    python3 ${template_path}/scripts/trace_compare.py \
      trace_sst_canonical.json trace_rtl_canonical.json \
      -o trace_diff.json || true
  else
    printf "  ${YELLOW}Warning:${NC} trace artifacts missing (SST='%s' VCD='%s'); skipping\n" \
      "${sst_trace}" "${vcd_file:-N/A}"
  fi
fi

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
