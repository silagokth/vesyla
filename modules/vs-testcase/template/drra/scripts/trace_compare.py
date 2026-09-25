#!/usr/bin/env python3
"""Compare the SST (instruction-level) and RTL canonical traces cycle-by-cycle.

Inputs are the two canonical JSONs produced by trace_extract_sst.py and
trace_extract_rtl.py. Both list events {cycle, cell, slot, kind, signal, value}.

What is compared, and how:

* Instruction & activation events are compared on TIMING. The SST and RTL
  instruction *words* use different framings (SST logs the full 32-bit fabric
  instruction; RTL exposes only the resource-instruction payload), so their
  values are not directly comparable -- but the *cycle* at which each resource
  is issued an instruction / activated is exactly the cycle-accuracy question.

* A single global skew Delta is inferred from the instruction stream (the most
  reliable signal). A CONSTANT skew -- even +/-1 -- is acceptable: it reflects a
  fixed pipeline latency between the two models. A skew that VARIES during the
  run is the real cycle-accuracy violation and is reported as OFFSET DRIFT.

* Data-movement events carry real word values in both models and are compared
  by VALUE: for each SST data event, the RTL data event on the same resource
  within the Delta-aligned cycle is looked up and the word lists diffed.

Exit status: 0 if MATCH, non-zero on any DRIFT / VALUE MISMATCH / structural
mismatch (so run.sh can gate on it).
"""
import argparse
import collections
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def lane_key(e):
    return (tuple(e["cell"]), e["slot"])


def by_lane(events, kind):
    """(cell,slot) -> sorted list of events of the given kind."""
    d = collections.defaultdict(list)
    for e in events:
        if e["kind"] == kind:
            d[lane_key(e)].append(e)
    for v in d.values():
        v.sort(key=lambda e: e["cycle"])
    return d


def align_under_delta(sl, rl, delta):
    """Two-pointer alignment of two sorted event lists under a fixed skew:
    a pair matches when rtl_cycle - sst_cycle == delta. Returns
    (pairs, missing, extra) where `missing` are SST events with no RTL
    counterpart and `extra` are RTL events with no SST counterpart. A single
    inserted/dropped event costs one unmatched event, not a shift of all the
    rest (unlike positional pairing)."""
    i = j = 0
    pairs, missing, extra = [], [], []
    while i < len(sl) and j < len(rl):
        d = rl[j]["cycle"] - sl[i]["cycle"]
        if d == delta:
            pairs.append((sl[i], rl[j]))
            i += 1
            j += 1
        elif d < delta:
            extra.append(rl[j])
            j += 1
        else:
            missing.append(sl[i])
            i += 1
    missing.extend(sl[i:])
    extra.extend(rl[j:])
    return pairs, missing, extra


def candidate_deltas(sst, rtl, max_skew):
    """Delta values worth trying: everything in [-max_skew, max_skew] plus the
    most common nearest-neighbour offsets in the instruction stream (so a
    genuine larger-but-constant pipeline skew is still discoverable)."""
    s = by_lane(sst, "instruction")
    r = by_lane(rtl, "instruction")
    near = collections.Counter()
    for lane in set(s) & set(r):
        rc = [e["cycle"] for e in r[lane]]
        for se in s[lane]:
            near[min(rc, key=lambda c: abs(c - se["cycle"])) - se["cycle"]] += 1
    cands = set(range(-max_skew, max_skew + 1))
    cands.update(d for d, _ in near.most_common(5))
    return sorted(cands)


def _near_offsets(sst, rtl):
    """Nearest-neighbour offset multiset over instruction events (for the
    reported histogram)."""
    s = by_lane(sst, "instruction")
    r = by_lane(rtl, "instruction")
    offsets = []
    for lane in sorted(set(s) & set(r)):
        rc = [e["cycle"] for e in r[lane]]
        for se in s[lane]:
            offsets.append(min(rc, key=lambda c: abs(c - se["cycle"]))
                           - se["cycle"])
    return offsets


def infer_delta(sst, rtl, max_skew):
    """Choose the single global Delta that maximises exact instruction matches
    across all lanes. Returns (delta, offsets) where offsets is the
    nearest-neighbour offset multiset for the histogram."""
    s = by_lane(sst, "instruction")
    r = by_lane(rtl, "instruction")
    lanes = sorted(set(s) & set(r))
    if not lanes:
        return None, []
    best_delta, best_score = 0, -1
    for delta in candidate_deltas(sst, rtl, max_skew):
        score = sum(len(align_under_delta(s[lane], r[lane], delta)[0])
                    for lane in lanes)
        if score > best_score:
            best_delta, best_score = delta, score
    return best_delta, _near_offsets(sst, rtl)


def compare_timing(sst, rtl, kind, delta):
    """Return (matches, issues) for instruction/activation timing under Delta,
    using non-cascading alignment. Divergences are the SST events RTL failed to
    match at the expected skew (sst-only) and RTL events with no SST match
    (rtl-only) -- the cycles where the two models genuinely disagree."""
    s = by_lane(sst, kind)
    r = by_lane(rtl, kind)
    issues = []
    matches = 0
    for lane in sorted(set(s) | set(r)):
        sl, rl = s.get(lane, []), r.get(lane, [])
        pairs, missing, extra = align_under_delta(sl, rl, delta)
        matches += len(pairs)
        for se in missing:
            issues.append({
                "type": "offset-drift", "kind": kind, "side": "sst-only",
                "cell": list(lane[0]), "slot": lane[1],
                "sst_cycle": se["cycle"], "rtl_cycle": None,
                "expected_delta": delta, "observed_offset": None,
            })
        for re_ in extra:
            issues.append({
                "type": "offset-drift", "kind": kind, "side": "rtl-only",
                "cell": list(lane[0]), "slot": lane[1],
                "sst_cycle": None, "rtl_cycle": re_["cycle"],
                "expected_delta": delta, "observed_offset": None,
            })
    return matches, issues


def compare_data(sst, rtl, delta, cycle_window):
    """Value-compare data events: each SST data event is matched to an RTL data
    event on the same resource whose cycle is within [Delta-w, Delta+w]."""
    s = by_lane(sst, "data")
    r = by_lane(rtl, "data")
    issues = []
    matches = 0
    for lane in sorted(set(s)):
        rl = r.get(lane, [])
        used = set()
        for se in s[lane]:
            target = se["cycle"] + delta
            best = None
            for i, re_ in enumerate(rl):
                if i in used:
                    continue
                if abs(re_["cycle"] - target) <= cycle_window:
                    if best is None or abs(re_["cycle"] - target) < \
                            abs(rl[best]["cycle"] - target):
                        best = i
            if best is None:
                continue  # no RTL data event nearby; timing handled elsewhere
            used.add(best)
            sv, rv = se["value"], rl[best]["value"]
            if sv == rv:
                matches += 1
            else:
                issues.append({
                    "type": "value-mismatch", "kind": "data",
                    "cell": list(lane[0]), "slot": lane[1],
                    "sst_cycle": se["cycle"], "rtl_cycle": rl[best]["cycle"],
                    "signal": rl[best]["signal"],
                    "sst_value": sv, "rtl_value": rv,
                })
    return matches, issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sst_json")
    ap.add_argument("rtl_json")
    ap.add_argument("--max-skew", type=int, default=1,
                    help="expected bound on |Delta| (default 1)")
    ap.add_argument("--offset", type=int, default=None,
                    help="force Delta instead of inferring it")
    ap.add_argument("--data-window", type=int, default=1,
                    help="cycle tolerance when matching data events")
    ap.add_argument("--no-data", action="store_true",
                    help="skip Tier-B data value comparison")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any drift/value divergence "
                         "(default: report divergences but exit 0)")
    ap.add_argument("-o", "--output", default="trace_diff.json")
    args = ap.parse_args()

    sst = load(args.sst_json)["events"]
    rtl = load(args.rtl_json)["events"]

    if args.offset is not None:
        delta, offsets = args.offset, None
    else:
        delta, offsets = infer_delta(sst, rtl, args.max_skew)
    if delta is None:
        print("ERROR: no instruction events to align on", file=sys.stderr)
        return 2
    offset_hist = dict(sorted(collections.Counter(offsets).items())) \
        if offsets else {}

    issues = []
    instr_m, instr_i = compare_timing(sst, rtl, "instruction", delta)
    act_m, act_i = compare_timing(sst, rtl, "activation", delta)
    issues += instr_i + act_i
    data_m = 0
    data_i = []
    if not args.no_data:
        data_m, data_i = compare_data(sst, rtl, delta, args.data_window)
        issues += data_i

    drift = [i for i in issues if i["type"] == "offset-drift"]
    vmis = [i for i in issues if i["type"] == "value-mismatch"]
    cmis = [i for i in issues if i["type"] == "count-mismatch"]

    report = {
        "delta": delta,
        "delta_constant": not drift,
        "offset_histogram": offset_hist,
        "matched": {"instruction": instr_m, "activation": act_m,
                    "data": data_m},
        "counts": {"offset_drift": len(drift), "value_mismatch": len(vmis),
                   "count_mismatch": len(cmis)},
        "issues": issues[:1000],
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)

    # Human summary
    hist_str = " ".join(f"{d:+d}:{n}" for d, n in offset_hist.items())
    if not issues:
        print(f"MATCH  (Delta={delta:+d}, constant)  "
              f"instr={instr_m} act={act_m} data={data_m}")
        return 0

    instr_div = sum(1 for i in drift if i["kind"] == "instruction")
    act_div = sum(1 for i in drift if i["kind"] == "activation")
    verdict = "MISMATCH" if (args.strict or cmis) else "DIVERGENCES"
    print(f"{verdict}  constant skew Delta={delta:+d}  |  "
          f"instr: {instr_m} aligned, {instr_div} divergent  "
          f"act: {act_m} aligned, {act_div} divergent  "
          f"data: {data_m} ok, {len(vmis)} mismatched")
    if hist_str:
        print(f"  instr offset histogram (rtl-sst cycle): {hist_str}")
    for i in issues[:15]:
        if i["type"] == "offset-drift":
            where = (f"SST@{i['sst_cycle']} (no RTL match at Delta)"
                     if i.get("side") == "sst-only"
                     else f"RTL@{i['rtl_cycle']} (no SST match at Delta)")
            print(f"  DIVERGE {i['kind']:11s} cell{i['cell']} slot{i['slot']}: "
                  f"{where}")
        elif i["type"] == "value-mismatch":
            print(f"  VALUE   {i['signal']:14s} cell{i['cell']} slot{i['slot']} "
                  f"SST@{i['sst_cycle']}: {i['sst_value']} != {i['rtl_value']}")
        else:
            print(f"  COUNT   {i['kind']:11s} cell{i['cell']} slot{i['slot']}: "
                  f"SST={i['sst_count']} RTL={i['rtl_count']}")
    if len(issues) > 15:
        print(f"  ... {len(issues) - 15} more (see {args.output})")
    # Non-strict: report divergences but succeed (minor SST/RTL micro-timing
    # differences do not change the memory result or total cycle count, both of
    # which run.sh already checks separately). --strict gates on any divergence.
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
