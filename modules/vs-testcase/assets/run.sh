#!/bin/sh
# Run the autotest suite. Do NOT use `set -e` here: pabot exits non-zero when any
# testcase fails, and we still want the summary step below to run in that case.
pabot --testlevelsplit -d output autotest_config.robot
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
