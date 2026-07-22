#!/usr/bin/env python3
"""Extract a canonical per-cycle event trace from the RTL simulator's VCD dump
(`trace.vcd`, produced by rtl_sim.sh in debug mode).

The VCD captures every resource interface port of every resource in every cell,
plus the testbench cycle counter `/fabric_tb/cycle_count` (added by rtl_sim.sh
specifically for this comparison). Interface signals live at scope

    fabric_tb/fabric_inst/cell_<R>_<C>_inst/resource_<S>_inst/<port>_<i>

where <port> is one of instr_en, instr, activate, word_data_in, word_data_out,
bulk_data_in, bulk_data_out, and <i> is the local index within a multi-slot
resource. The absolute slot is  S + i.

Questa emits multi-bit ports bit-blasted, one 1-bit `$var` per bit named
"<port>_<i> [b]"; this script reconstructs the integer value. The cycle axis is
recovered from cycle_count (an int reg whose value is the current cycle).

Output matches trace_extract_sst.py's schema:
  {
    "meta": {"source": "rtl", "cycles": <int>, "word_bitwidth": <int>},
    "events": [ {"cycle","cell":[r,c],"slot","kind","signal","value"}, ... ]
  }
Emitted events (sparse, mirroring the SST event semantics):
  instr    : at every cycle where instr_en_<i> == 1     value = instr word (int)
  activate : on rising transition of activate_<i> to !=0 value = mask (int)
  data     : when a data bus changes to a new nonzero    value = [words]
             (word_data_*/bulk_data_*, little-endian, word_bitwidth-bit words)
"""
import argparse
import json
import re
import sys

PORT_RE = re.compile(
    r"^(instr_en|instr|activate|word_data_in|word_data_out|"
    r"bulk_data_in|bulk_data_out)_(\d+)$"
)
SCOPE_RE = re.compile(r"cell_(\d+)_(\d+)_inst")
RES_RE = re.compile(r"resource_(\d+)_inst")
CYCLE_PATH = ("fabric_tb", "cycle_count")


class Sig:
    # A resource interface port is keyed by (cell, base_slot, local, port).
    # `base_slot` is the resource_<S>_inst slot; `local` is the per-slot index
    # within a multi-slot resource. Events are attributed to `base_slot` to
    # match the SST model, which logs every port of a resource under the
    # resource's base slot_id.
    __slots__ = ("cell", "base_slot", "local", "port", "bits", "scalar_id",
                 "width")

    def __init__(self, cell, base_slot, local, port):
        self.cell = cell
        self.base_slot = base_slot
        self.local = local
        self.port = port
        self.bits = {}          # bit_index -> current 0/1
        self.scalar_id = None   # for width-1 ports declared as scalar
        self.width = 1

    def value(self):
        if self.scalar_id is not None:
            return self.bits.get(0, 0)
        if not self.bits:
            return 0
        return sum((v & 1) << b for b, v in self.bits.items())


def parse_header(f):
    """Return (id2sig, id2isbit, cycle_id). Advances f past $enddefinitions."""
    scope = []
    sigs = {}            # (cell,slot,port) -> Sig
    id2targets = {}      # vcd id -> list of (sig, bit_index or None)
    cycle_id = None
    for line in f:
        s = line.strip()
        if s.startswith("$scope"):
            scope.append(s.split()[2])
        elif s.startswith("$upscope"):
            if scope:
                scope.pop()
        elif s.startswith("$var"):
            m = re.match(r"\$var\s+\w+\s+(\d+)\s+(\S+)\s+(.+?)\s+\$end", s)
            if not m:
                continue
            width, ident, name = int(m.group(1)), m.group(2), m.group(3)
            # cycle_count (top of tb)
            base = re.sub(r"\s*\[\d+(:\d+)?\]$", "", name)
            if tuple(scope[-2:]) == CYCLE_PATH or (
                scope and scope[-1] == "fabric_tb" and base == "cycle_count"
            ):
                if base == "cycle_count":
                    cycle_id = ident
                    continue
            # resource interface ports
            pm = PORT_RE.match(base)
            if not pm:
                continue
            # locate cell + resource scope in the path
            cell = res_slot = None
            for sc in scope:
                cm = SCOPE_RE.match(sc)
                if cm:
                    cell = (int(cm.group(1)), int(cm.group(2)))
                rm = RES_RE.match(sc)
                if rm:
                    res_slot = int(rm.group(1))
            if cell is None or res_slot is None:
                continue
            port, local = pm.group(1), int(pm.group(2))
            key = (cell, res_slot, local, port)
            sig = sigs.get(key)
            if sig is None:
                sig = sigs[key] = Sig(cell, res_slot, local, port)
            # bit index?
            bm = re.search(r"\[(\d+)\]$", name)
            if bm:
                bit = int(bm.group(1))
                sig.width = max(sig.width, bit + 1)
                id2targets.setdefault(ident, []).append((sig, bit))
            else:
                sig.scalar_id = ident
                sig.width = max(sig.width, width)
                id2targets.setdefault(ident, []).append((sig, 0))
        elif s.startswith("$enddefinitions"):
            break
    return sigs, id2targets, cycle_id


def to_words(value, width, word_bitwidth):
    n = max(1, (width + word_bitwidth - 1) // word_bitwidth)
    mask = (1 << word_bitwidth) - 1
    return [(value >> (i * word_bitwidth)) & mask for i in range(n)]


def extract(vcd_path, word_bitwidth):
    with open(vcd_path) as f:
        sigs, id2targets, cycle_id = parse_header(f)
        if cycle_id is None:
            print("WARN: cycle_count not found in VCD; cycles will be 0",
                  file=sys.stderr)
        cycle = 0
        max_cycle = 0
        # snapshots[cycle][(cell,slot,port)] = value  (end-of-cycle state)
        snapshots = {}

        def snapshot():
            row = snapshots.setdefault(cycle, {})
            for key, sig in sigs.items():
                row[key] = sig.value()

        for line in f:
            s = line.strip()
            if not s:
                continue
            c0 = s[0]
            if c0 == "#":
                # time step: before advancing, the values belong to `cycle`
                continue
            if c0 in "01xzXZ":
                ident = s[1:]
                val = 1 if c0 == "1" else 0
                if ident == cycle_id:
                    continue
                for sig, bit in id2targets.get(ident, ()):
                    sig.bits[bit] = val
            elif c0 in "bB":
                parts = s.split()
                if len(parts) != 2:
                    continue
                bitstr, ident = parts[0][1:], parts[1]
                intval = int(bitstr.replace("x", "0").replace("z", "0"), 2) \
                    if bitstr else 0
                if ident == cycle_id:
                    new_cycle = intval
                    if new_cycle != cycle:
                        snapshot()  # flush the cycle that is ending
                        cycle = new_cycle
                        max_cycle = max(max_cycle, cycle)
                    continue
                for sig, bit in id2targets.get(ident, ()):
                    if len(id2targets.get(ident, ())) == 1 and \
                            sig.scalar_id == ident:
                        # packed vector on a scalar-registered id
                        sig.bits = {i: (intval >> i) & 1
                                    for i in range(sig.width)}
                    else:
                        sig.bits[bit] = intval & 1
            elif c0 in "rR":
                continue
        snapshot()  # final cycle

    # Build sparse events from per-cycle snapshots. Each event is attributed to
    # the resource base slot (to match SST), iterating over every local port.
    events = []
    cycles = sorted(snapshots)
    prev = {}
    # unique (cell, base_slot, local) lanes present
    lanes = sorted({(c, b, l) for (c, b, l, _) in sigs})
    widths = {k: s.width for k, s in sigs.items()}
    for cyc in cycles:
        row = snapshots[cyc]
        for (cell, base, local) in lanes:
            def g(port, r=row, cell=cell, base=base, local=local):
                return r.get((cell, base, local, port), 0)

            if g("instr_en"):
                events.append({"cycle": cyc, "cell": list(cell), "slot": base,
                               "kind": "instruction", "signal": "instr",
                               "value": g("instr")})
            act = g("activate")
            if act and act != prev.get((cell, base, local, "activate"), 0):
                events.append({"cycle": cyc, "cell": list(cell), "slot": base,
                               "kind": "activation", "signal": "activate",
                               "value": act})
            for port in ("word_data_out", "bulk_data_out",
                         "word_data_in", "bulk_data_in"):
                v = g(port)
                if v and v != prev.get((cell, base, local, port)):
                    events.append({"cycle": cyc, "cell": list(cell),
                                   "slot": base, "kind": "data",
                                   "signal": port,
                                   "value": to_words(
                                       v, widths[(cell, base, local, port)],
                                       word_bitwidth)})
        for key in sigs:
            prev[key] = row.get(key, 0)

    events.sort(key=lambda e: (e["cycle"], e["cell"], e["slot"], e["signal"]))
    return {"meta": {"source": "rtl", "cycles": max_cycle,
                     "word_bitwidth": word_bitwidth},
            "events": events}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vcd", help="path to trace.vcd")
    ap.add_argument("-o", "--output", default="rtl_canonical.json")
    ap.add_argument("--word-bitwidth", type=int, default=16)
    args = ap.parse_args()
    canon = extract(args.vcd, args.word_bitwidth)
    with open(args.output, "w") as f:
        json.dump(canon, f, indent=1)
    print(f"RTL canonical: {len(canon['events'])} events, "
          f"{canon['meta']['cycles']} cycles -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
