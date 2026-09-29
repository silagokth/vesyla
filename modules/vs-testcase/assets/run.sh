#!/bin/sh
# Run the autotest suite. Do NOT use `set -e` here: pabot exits non-zero when any
# testcase fails, and we still want the summary step below to run in that case.
#
# Usage: ./run.sh [--frontend=pasm|mlir|all] [pabot/robot options...]
# Every testcase has a pasm variant, and an mlir variant when it carries
# mlir/*.mlir; --frontend picks which of them run (default: pasm). Any other
# arguments are passed through to pabot, e.g. --include drra::basic.
frontend=pasm
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
  -h | --help)
    echo "Usage: $0 [--frontend=pasm|mlir|all] [pabot/robot options...]"
    echo "  --frontend=pasm   run the testcases' pasm/*.pasm programs (default)"
    echo "  --frontend=mlir   run the testcases' mlir/*.mlir programs"
    echo "  --frontend=all    run both"
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

# $frontend_filter is deliberately unquoted: it is either empty or two words.
pabot --testlevelsplit -d output $frontend_filter "$@" autotest_config.robot
pabot_rc=$?

# Summarize which step each failing testcase reached, using the per-test failure
# message Robot Framework records in output/output.xml. Without this, pabot only
# prints "FAILED <suite>.<test>" on the console and the step reason (Setup / SST
# run / SST output / RTL run / RTL output) stays buried in report.html.
if [ -f output/output.xml ]; then
  python3 - <<'PY'
import re, xml.etree.ElementTree as ET

root = ET.parse("output/output.xml").getroot()
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
