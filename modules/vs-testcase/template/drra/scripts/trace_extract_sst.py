#!/usr/bin/env python3
"""Extract a canonical per-cycle event trace from the SST instruction-level
simulator's Chrome-trace output (`trace_complete.json`).

The SST model (drra-components) logs discrete events with:
  - ts   : raw SST cycle counter (the SST clock runs at 10x the real clock,
           so the real cycle is ts // 10)
  - tid  : an integer encoding base + row + col + slot, zero-padded to 10
           digits, i.e.  <base><row(3)><col(3)><slot(3)>.
           base 1 = resource lane, base 2 = controller lane.
  - name : event name  (e.g. "instruction", "activation", "rf_write_wide",
           "iosram_write_bulk", ...)
  - args : event-specific payload. Notable keys:
             instruction        -> {instruction, instruction_bin,
                                     instruction_hex, pc}
             activation         -> {ports}
             data movements     -> {address, data:"[w0, w1, ...]", size}

This script projects those events onto a canonical schema shared with the RTL
extractor so the two can be diffed cycle-by-cycle:

  {
    "meta": {"source": "sst", "cycles": <int>, "word_bitwidth": <int>},
    "events": [
       {"cycle": <int>, "cell": [row, col], "slot": <int>,
        "kind": "instruction"|"activation"|"data",
        "signal": <canonical signal name>,
        "value": <int | [int,...] | str>},
       ...
    ]
  }

Canonical signal names (must match trace_extract_rtl.py):
  instruction  -> "instr"     value = instruction word (int, from hex)
  activation   -> "activate"  value = ports bitmask (int)
  data move    -> "data"      value = [words]  (decimal, little-endian)
"""
import argparse
import json
import re
import sys


def decode_tid(tid):
    """<base><row(3)><col(3)><slot(3)> -> (base, row, col, slot)."""
    s = str(tid).rjust(10, "0")
    base = int(s[0])
    row = int(s[1:4])
    col = int(s[4:7])
    slot = int(s[7:10])
    return base, row, col, slot


def parse_words(data_str):
    """'[2, 1, 0]' -> [2, 1, 0]."""
    inner = data_str.strip().lstrip("[").rstrip("]").strip()
    if not inner:
        return []
    return [int(x) for x in re.split(r"[,\s]+", inner) if x != ""]


# Event-name -> (kind, canonical signal). Data-movement events all map to the
# generic "data" signal; the concrete name is kept in `event` for debugging.
DATA_EVENT_RE = re.compile(
    r"(read|write|bulk|dsu|from_io|to_io|from_sram|to_sram|input|output)", re.I
)


def classify(name):
    if name == "instruction":
        return "instruction", "instr"
    if name == "activation":
        return "activation", "activate"
    if DATA_EVENT_RE.search(name):
        return "data", "data"
    return None, None


def extract(trace_path):
    with open(trace_path) as f:
        doc = json.load(f)
    events_in = doc.get("traceEvents", doc if isinstance(doc, list) else [])

    events = []
    max_cycle = 0
    for ev in events_in:
        if ev.get("ph") not in ("X", "B"):  # skip metadata (M) and E closers
            continue
        if "tid" not in ev or "ts" not in ev:
            continue
        base, row, col, slot = decode_tid(ev["tid"])
        if base != 1:  # resource lane only for signal comparison
            continue
        cycle = int(ev["ts"]) // 10
        max_cycle = max(max_cycle, cycle)
        kind, signal = classify(ev.get("name", ""))
        if kind is None:
            continue
        args = ev.get("args", {})
        if kind == "instruction":
            hexv = args.get("instruction_hex")
            value = int(hexv, 16) if hexv else None
        elif kind == "activation":
            p = args.get("ports")
            value = int(p) if p is not None else None
        else:  # data
            value = parse_words(args.get("data", "[]"))
        events.append(
            {
                "cycle": cycle,
                "cell": [row, col],
                "slot": slot,
                "kind": kind,
                "signal": signal,
                "event": ev.get("name"),
                "value": value,
            }
        )

    events.sort(key=lambda e: (e["cycle"], e["cell"], e["slot"], e["signal"]))
    return {
        "meta": {"source": "sst", "cycles": max_cycle},
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_json", help="path to trace_complete.json")
    ap.add_argument("-o", "--output", default="sst_canonical.json")
    args = ap.parse_args()
    canon = extract(args.trace_json)
    with open(args.output, "w") as f:
        json.dump(canon, f, indent=1)
    print(
        f"SST canonical: {len(canon['events'])} events, "
        f"{canon['meta']['cycles']} cycles -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
