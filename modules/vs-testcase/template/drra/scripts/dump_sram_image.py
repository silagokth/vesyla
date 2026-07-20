#!/usr/bin/env python3
"""Best-effort dump of a sram_image .bin file to .hex and .txt siblings.

The .bin format is one line per memory row: "<addr> <0/1-bits>".
Produces:
  <stem>.hex  -> "<addr> <hex>"
  <stem>.txt  -> "<addr> [v0, v1, ...]"  (values decoded as <data-type>)

Never raises; missing input or malformed lines are skipped.
"""

import argparse
import os
import re
import sys

LINE_RE = re.compile(r"\s*(\d+)\s+([01]+)\s*")

DTYPES = {
    "int8_t":   (8,  True),
    "uint8_t":  (8,  False),
    "int16_t":  (16, True),
    "uint16_t": (16, False),
    "int32_t":  (32, True),
    "uint32_t": (32, False),
    "int64_t":  (64, True),
    "uint64_t": (64, False),
}


def warn(msg):
    print(f"[dump_sram_image] {msg}", file=sys.stderr)


def bits_to_hex(bits):
    pad = (-len(bits)) % 4
    bits = "0" * pad + bits
    return "".join(f"{int(bits[i:i+4], 2):x}" for i in range(0, len(bits), 4))


def bits_to_value(chunk, signed):
    v = int(chunk, 2)
    if signed and chunk[0] == "1":
        v -= 1 << len(chunk)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin_file")
    ap.add_argument("--data-type", default="int16_t",
                    help=f"one of {sorted(DTYPES)} (default: int16_t)")
    args = ap.parse_args()

    bin_path = args.bin_file
    if not os.path.isfile(bin_path):
        warn(f"input not found, skipping: {bin_path}")
        return 0

    width, signed = DTYPES.get(args.data_type, DTYPES["int16_t"])
    if args.data_type not in DTYPES:
        warn(f"unknown data type '{args.data_type}', falling back to int16_t")

    stem, ext = os.path.splitext(bin_path)
    if ext != ".bin":
        stem = bin_path
    hex_path = stem + ".hex"
    txt_path = stem + ".txt"

    try:
        with open(bin_path, "r") as f:
            lines = f.readlines()
    except OSError as e:
        warn(f"cannot read {bin_path}: {e}")
        return 0

    hex_out = []
    txt_out = []
    for raw in lines:
        m = LINE_RE.match(raw)
        if not m:
            continue
        addr, bits = m.group(1), m.group(2)
        try:
            hex_out.append(f"{addr} {bits_to_hex(bits)}")
        except Exception:
            pass
        if len(bits) % width == 0:
            try:
                vals = [bits_to_value(bits[i:i+width], signed)
                        for i in range(0, len(bits), width)]
                txt_out.append(f"{addr} [{', '.join(str(v) for v in vals)}]")
            except Exception:
                pass

    try:
        with open(hex_path, "w") as f:
            f.write("\n".join(hex_out) + ("\n" if hex_out else ""))
    except OSError as e:
        warn(f"cannot write {hex_path}: {e}")

    try:
        with open(txt_path, "w") as f:
            f.write("\n".join(txt_out) + ("\n" if txt_out else ""))
    except OSError as e:
        warn(f"cannot write {txt_path}: {e}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        warn(f"unexpected error, ignoring: {e}")
        sys.exit(0)
