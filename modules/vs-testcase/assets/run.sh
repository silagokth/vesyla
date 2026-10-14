#!/bin/sh
# Run the autotest suite. Do NOT use `set -e` here: pabot exits non-zero when any
# testcase fails, and we still want the summary step below to run in that case.
#
# Options: see the USAGE text below (./run.sh --help). `vesyla testcase generate`
# prints the same text, so keep it in that heredoc.
#
# Every testcase has a pasm variant, and an mlir variant when it carries
# mlir/*.mlir; --frontend picks which of them run by their frontend:<name> tag.
# --rtl-target is handed to the testcases' rtl_sim.sh through
# VESYLA_RTL_TARGETS, which adds it to the Bender targets on top of `sim`.
# --no-cleanup keeps the work/ directory of every testcase; without it only the
# failing ones are kept.
# --no-reconfiguration builds every testcase's RTL without reconfiguration
# (the testcases' run.sh -nr); model 2 is skipped then.
frontend=pasm
cleanup_flag=""
noreconf_flag=""
rtl_targets="${VESYLA_RTL_TARGETS:-}"
while [ $# -gt 0 ]; do
  case "$1" in
  --frontend=*)
    frontend="${1#*=}"
    shift
    ;;
  --frontend)
    frontend="$2"
    shift 2
    ;;
  --rtl-target=*)
    rtl_targets="$rtl_targets ${1#*=}"
    shift
    ;;
  --rtl-target)
    rtl_targets="$rtl_targets $2"
    shift 2
    ;;
  --no-cleanup)
    cleanup_flag="--variable NO_CLEANUP:True"
    shift
    ;;
  -nr | --no-reconfiguration)
    noreconf_flag="--variable NO_RECONFIGURATION:True"
    shift
    ;;
  -h | --help)
    cat <<'USAGE'
Usage: ./run.sh [options] [pabot/robot options...]
  --frontend=pasm          run the testcases' pasm/*.pasm programs (default)
  --frontend=mlir          run the testcases' mlir/*.mlir programs
  --frontend=all           run both
  --rtl-target=TARGET      add a Bender target to the RTL simulation on top of sim,
                           e.g. freepdk45 for the OpenRAM macro models; repeatable
                           (default: sim only, flip-flop memories)
  --no-cleanup             keep every testcase's work/<testcase>_<id>/ directory
                           (default: only failing testcases keep theirs)
  -nr, --no-reconfiguration
                           build the RTL without reconfiguration: the controllers
                           only issue the activations, the resources hardcode the
                           rest of the program; model 2 (SST) is skipped
  -h, --help               show this help
Any other argument goes to pabot, e.g. --include <testcase group> to run one group.
USAGE
    exit 0
    ;;
  *)
    break
    ;;
  esac
done

case "$frontend" in
pasm | mlir) frontend_filter="--include frontend:$frontend" ;;
all) frontend_filter="" ;;
*)
  echo "ERROR: unknown --frontend '$frontend' (choose from pasm, mlir, all)" >&2
  exit 1
  ;;
esac

# Trim the leading space the loop above leaves.
rtl_targets="$(echo $rtl_targets)"
export VESYLA_RTL_TARGETS="$rtl_targets"
echo "RTL simulation targets: sim${rtl_targets:+ $rtl_targets}"
if [ -n "$cleanup_flag" ]; then
  echo "Keeping every testcase's work directory under work/"
fi
if [ -n "$noreconf_flag" ]; then
  echo "Building the RTL without reconfiguration"
fi

# $frontend_filter, $cleanup_flag and $noreconf_flag are deliberately unquoted:
# each is either empty or two words.
pabot --testlevelsplit -d output $frontend_filter $cleanup_flag $noreconf_flag "$@" autotest_config.robot
pabot_rc=$?

# Summarize which step each failing testcase reached, using the per-test failure
# message Robot Framework records in output/output.xml. Without this, pabot only
# prints "FAILED <suite>.<test>" on the console and the step reason (Setup / SST
# run / SST output / RTL run / RTL output) stays buried in report.html.
if [ -f output/output.xml ]; then
  python3 - <<'PY'
import re, xml.etree.ElementTree as ET

root = ET.parse("output/output.xml").getroot()

# The cycle counts of the passing testcases: the message the suite's Report
# Cycles keyword gives them ("Cycles: sst=.. rtl=.. start_offset=..").
cycles = []
for test in root.iter("test"):
    status = test.find("status")
    msg = (status.text or "").strip() if status is not None else ""
    if status is not None and status.get("status") == "PASS" and msg.startswith("Cycles:"):
        values = dict(kv.split("=", 1) for kv in msg[len("Cycles:"):].split() if "=" in kv)
        cycles.append((test.get("name"), values))
if cycles:
    keys = [k for k in ("sst", "rtl", "start_offset") if any(k in v for _, v in cycles)]
    width = max(len(name) for name, _ in cycles)
    print("\n===== CYCLE COUNTS =====")
    print("  %-*s %s" % (width, "testcase", "".join("%14s" % k for k in keys)))
    for name, values in sorted(cycles):
        print("  %-*s %s" % (width, name, "".join("%14s" % values.get(k, "-") for k in keys)))

failed = []
for test in root.iter("test"):
    status = test.find("status")
    if status is not None and status.get("status") == "FAIL":
        msg = (status.text or "").strip()
        # Drop Robot's trailing value dump (e.g. ": 3 == 3") for readability.
        msg = re.sub(r":\s*-?\d+\s*(==|!=)\s*-?\d+\s*$", "", msg).strip()
        failed.append((test.get("name"), msg or "Unknown failure"))

if failed:
    print("\n===== FAILED TESTCASES (step that failed) =====")
    for name, msg in failed:
        print("  ✗ %s\n      -> %s" % (name, msg))
    print("\n%d testcase(s) failed. Full logs: output/report.html" % len(failed))
PY
fi

exit $pabot_rc
