#!/usr/bin/env bash
#
# lint_fabric.sh — Tier A synthesizability check: lint + elaborate an assembled
# DRRA fabric with OPEN tools only. No PDK, no liberty, no server — safe to run
# on a stock GitHub-hosted runner on every PR.
#
# What it catches (PDK-independent, the bulk of "did this break synthesis"):
#   * verible-verilog-syntax — parse breakage from templating / bad SV (hard).
#   * verible-verilog-lint   — style / a few synthesis-relevant rules (soft;
#                              hard when STRICT=1).
#   * slang elaboration      — full SystemVerilog elaboration of the `fabric`
#                              top, natively handling the agu_cfg_if interfaces
#                              that plain Yosys cannot. Flags unresolved modules,
#                              port/width mismatches, and — because the RTL is
#                              written with always_comb — incomplete-assignment
#                              latches. Hard fail on errors (when slang present).
#
# slang is treated as best-effort: if the binary is not on PATH the check is
# skipped with a notice (verible remains the guaranteed baseline). Install slang
# in the workflow to activate elaboration + latch checks.
#
# Usage: lint_fabric.sh <fabric_rtl_root>      # e.g. build/rtl
#   env TOP    top module name      (default: fabric)
#   env STRICT 1 => verible lint warnings are fatal too (default: 0)
#
set -euo pipefail

RTL_ROOT="${1:?usage: lint_fabric.sh <fabric_rtl_root>}"
TOP="${TOP:-fabric}"
STRICT="${STRICT:-0}"

# Build the synthesizable file set. `vesyla component assemble` copies the whole
# common/ library (incl. all three sram.{sim,gf22,tsmc28}.sv variants, which each
# define `module sram`) and the testbenches, regardless of what the fabric
# actually instantiates. For a no-SRAM fabric the sram macro is never wired in,
# so exclude it (keeping it would feed slang three duplicate `sram` modules).
# Also exclude testbenches and the AGU self-test files (sim-only, not synth).
mapfile -t SV < <(find "$RTL_ROOT" -type f -name '*.sv' \
  -not -path '*/common/sram/*' \
  -not -path '*/test/*' \
  -not -path '*/tb/*' | sort)
if [[ ${#SV[@]} -eq 0 ]]; then
  echo "ERROR: no .sv files under $RTL_ROOT" >&2
  exit 2
fi
echo "==> ${#SV[@]} SystemVerilog files (top=$TOP, sram/tb/test excluded)"

# Stay stdcell-only: assert the fabric cone does not *instantiate* the sram
# macro (an iosram fabric would, and that needs the confidential GF22 SRAM lib).
if grep -lE '^[[:space:]]*sram[[:space:]]+(#|[A-Za-z_])' "${SV[@]}" >/dev/null 2>&1; then
  echo "ERROR: fabric instantiates the sram macro — not stdcell-only." >&2
  echo "       Tier A expects a no-SRAM (io) fabric." >&2
  exit 4
fi

fail=0

# 1. Verible syntax — hard gate (catches templating/parse breakage). -----------
if command -v verible-verilog-syntax >/dev/null 2>&1; then
  echo "== verible-verilog-syntax =="
  verible-verilog-syntax "${SV[@]}" || fail=1
else
  echo "WARN: verible-verilog-syntax not found; skipping syntax check"
fi

# 2. Verible lint — soft by default (style-heavy), fatal under STRICT. ----------
if command -v verible-verilog-lint >/dev/null 2>&1; then
  echo "== verible-verilog-lint =="
  rules=()
  [[ -f "$(dirname "$0")/verible.rules" ]] && rules=(--rules_config "$(dirname "$0")/verible.rules")
  if verible-verilog-lint "${rules[@]}" "${SV[@]}"; then
    echo "lint: clean"
  else
    if [[ "$STRICT" == "1" ]]; then fail=1; else echo "(lint warnings — non-fatal; set STRICT=1 to enforce)"; fi
  fi
else
  echo "WARN: verible-verilog-lint not found; skipping lint"
fi

# 3. slang elaboration — hard gate when present (interfaces, latches, widths). --
if command -v slang >/dev/null 2>&1; then
  echo "== slang elaborate (top=$TOP) =="
  # --top forces hierarchy elaboration from `fabric`; errors -> non-zero exit.
  slang --top "$TOP" --error-limit=100 "${SV[@]}" || fail=1
else
  echo "WARN: slang not found; skipping elaboration (latch/interface/width checks)."
  echo "      Install slang in the workflow to enable the full Tier A gate."
fi

if [[ "$fail" == "0" ]]; then
  echo "Tier A synthesizability lint PASSED"
else
  echo "Tier A synthesizability lint FAILED"
  exit 1
fi
