#!/usr/bin/env python3
"""Render a schedule timetable JSON (produced by ScheduleEpochPass) as an SVG
Gantt chart.

Input JSON layout:
    {
      "epoch": "rb1",
      "total_latency": 95,
      "resources": [
        { "row": 0, "col": 0, "slot": 0,
          "operations": [
            { "name": "swb", "port": 0, "start": 0, "end": 93, "duration": 93 }
          ]
        }
      ],
      "controllers": [
        { "row": 0, "col": 0,
          "instructions": [
            { "type": "act", "cycle": 1, "end": 2 },
            { "type": "conf", "cycle": 2, "end": 3, "slot": 4, "port": 0 }
          ]
        }
      ]
    }

One lane per resource (row, col, slot), each operation a bar from start to end.
Each cell's controller (row, col) gets its own lane directly above its slot
lanes; each controller instruction is a bar over [cycle, end) — most span a
single cycle, while a wait spans its full N+1 cycles as a block. A control
instruction that names a target slot (evt/conf/rep/trans, ...) is also echoed as
a dashed box on that slot's resource lane. The x-axis is cycles. Only the Python standard
library is used and the output SVG needs no external renderer.

Usage:
    timetable.py <file-or-dir> [<file-or-dir> ...] [-o OUTPUT_DIR]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

# Geometry (pixels).
LEFT_MARGIN = 170
TOP_MARGIN = 56
RIGHT_MARGIN = 24
BOTTOM_MARGIN = 24
LANE_HEIGHT = 30
BAR_PAD = 4
CELL_WIDTH = 22
MIN_BAR_WIDTH = 6

# Operation bar colors, cycled by a stable hash of the operation name so the
# same operation keeps its color across epochs.
PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def esc(text):
    """Escape a string for inclusion in XML/SVG text."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def color_for(name):
    return PALETTE[hash(name) % len(PALETTE)]


def lane_label(resource):
    return "({},{}) slot {}".format(
        resource.get("row", "?"), resource.get("col", "?"),
        resource.get("slot", "?")
    )


# Approximate width of a label character at font-size 11, used to size the
# variable-width cycle columns so abbreviated labels like "2,...,5" fit.
CHAR_W = 6.6


def op_bounds(op):
    s = int(op.get("start", 0))
    e = int(op.get("end", s + 1))
    if e <= s:
        e = s + 1
    return s, e


def ctrl_bounds(instr):
    """A controller instruction spans [cycle, end); most occupy a single cycle,
    while a wait spans its full N+1 cycles."""
    s = int(instr.get("cycle", 0))
    e = int(instr.get("end", s + 1))
    if e <= s:
        e = s + 1
    return s, e


# Control instruction types that configure a resource (as opposed to act/wait).
CONFIG_TYPES = {"conf", "rep", "evt", "trans"}


def _cycle_set(intervals, total):
    """Union of half-open [s, e) cycle ranges, clipped to [0, total)."""
    cycles = set()
    for s, e in intervals:
        lo, hi = max(0, s), (min(total, e) if total else e)
        if hi > lo:
            cycles.update(range(lo, hi))
    return cycles


def compute_utilization(data):
    """Per-resource cycle accounting over [0, total_latency).

    acting = cycles with any ROP active on the resource; config = cycles with a
    conf/rep/evt/trans targeting it; active = their union; stall = total - active.
    acting and config are independent raw counts and may overlap. Percentages are
    relative to total_latency. Only slots that appear as resources are reported."""
    total = int(data.get("total_latency", 0) or 0)

    # Config cycles per (row, col, slot) from the controller streams.
    config_by_res = {}
    for c in data.get("controllers", []):
        rc = (c.get("row", 0), c.get("col", 0))
        for instr in c.get("instructions", []):
            if instr.get("type") in CONFIG_TYPES and "slot" in instr:
                key = (rc[0], rc[1], instr["slot"])
                config_by_res.setdefault(key, []).append(ctrl_bounds(instr))

    def pct(n):
        return round(100 * n / total) if total else 0

    util = []
    for res in data.get("resources", []):
        key = (res.get("row", 0), res.get("col", 0), res.get("slot", 0))
        acting = _cycle_set(
            (op_bounds(op) for op in res.get("operations", [])), total)
        config = _cycle_set(config_by_res.get(key, []), total)
        active = acting | config
        stall = max(total - len(active), 0)
        util.append({
            "row": key[0], "col": key[1], "slot": key[2], "total": total,
            "acting": len(acting), "acting_pct": pct(len(acting)),
            "config": len(config), "config_pct": pct(len(config)),
            "active": len(active), "active_pct": pct(len(active)),
            "stall": stall, "stall_pct": pct(stall),
        })
    util.sort(key=lambda u: (u["row"], u["col"], u["slot"]))
    return util


def util_label(u):
    """Compact per-resource utilization annotation for the chart."""
    return "act {} ({}%)  cfg {} ({}%)  stall {} ({}%)".format(
        u["acting"], u["acting_pct"], u["config"], u["config_pct"],
        u["stall"], u["stall_pct"])


def build_lanes(resources, controllers, util_map):
    """Order lanes cell by cell: each cell's controller lane (if any) first, then
    its slot (resource) lanes, so the controller sits atop the ROPs it drives.

    All ops are normalized to {start, end, label, ...} so a single op_bounds
    reads them. A control instruction (evt/conf/rep) that names a target slot is
    additionally echoed as a box on that slot's resource lane, carrying the same
    type/cycle info, so the target of each control instruction is visible. Each
    resource lane is tagged with its utilization stats (util_map) for annotation."""
    cells = sorted(set(
        [(r.get("row", 0), r.get("col", 0)) for r in resources]
        + list(controllers.keys())
    ))

    def norm_res_op(op):
        s, e = op_bounds(op)
        return {"start": s, "end": e, "label": op.get("name", ""),
                "port": op.get("port", ""),
                "duration": op.get("duration", e - s)}

    def norm_ctrl(instr):
        s, e = ctrl_bounds(instr)
        return {"start": s, "end": e, "label": instr.get("type", "")}

    lanes = []
    for cell in cells:
        cell_ctrl = controllers.get(cell, [])
        if cell in controllers:
            lanes.append({
                "label": "({},{}) ctrl".format(*cell),
                "ops": [norm_ctrl(i) for i in cell_ctrl],
            })
        cell_res = sorted(
            (r for r in resources
             if (r.get("row", 0), r.get("col", 0)) == cell),
            key=lambda r: r.get("slot", 0),
        )
        for res in cell_res:
            slot = res.get("slot", 0)
            ops = [norm_res_op(op) for op in res.get("operations", [])]
            # Echo control instructions targeting this slot as boxes on the
            # resource lane (same type/cycle as in the controller lane).
            for instr in cell_ctrl:
                if instr.get("slot") == slot:
                    box = norm_ctrl(instr)
                    box["target"] = True
                    box["port"] = instr.get("port", "")
                    ops.append(box)
            lanes.append({
                "label": lane_label(res),
                "ops": ops,
                "util": util_map.get((cell[0], cell[1], slot)),
            })
    return lanes


def render(data, utilization=None):
    epoch = data.get("epoch", "epoch")
    total = int(data.get("total_latency", 0) or 0)
    resources = data.get("resources", [])
    controllers = {
        (c.get("row", 0), c.get("col", 0)): c.get("instructions", [])
        for c in data.get("controllers", [])
    }

    if utilization is None:
        utilization = compute_utilization(data)
    util_map = {(u["row"], u["col"], u["slot"]): u for u in utilization}

    lanes = build_lanes(resources, controllers, util_map)

    # Span must cover the latency and the latest event end (ROP or controller).
    span = max(total, 1)
    for lane in lanes:
        for op in lane["ops"]:
            span = max(span, op_bounds(op)[1])

    # Assign overlapping operations within a lane to separate sub-rows so the
    # overlap is visible (greedy interval partitioning by start time). Each lane
    # then occupies one row per track.
    def assign_tracks(ops):
        track_end = []
        track_of = [0] * len(ops)
        for i in sorted(range(len(ops)), key=lambda j: op_bounds(ops[j])):
            s, e = op_bounds(ops[i])
            placed = False
            for t in range(len(track_end)):
                if track_end[t] <= s:
                    track_of[i] = t
                    track_end[t] = e
                    placed = True
                    break
            if not placed:
                track_of[i] = len(track_end)
                track_end.append(e)
        return track_of, max(len(track_end), 1)

    lane_tracks = [assign_tracks(lane["ops"]) for lane in lanes]
    total_rows = max(sum(nt for (_, nt) in lane_tracks), 1)

    # Content only changes at an operation start or end, so consecutive cycles
    # between two such events are identical. Collapse each maximal run of
    # equal-content cycles into a single column [b_i, b_{i+1}). Every operation
    # interval aligns to these boundaries, so each op spans whole columns.
    boundaries = {0, span}
    for lane in lanes:
        for op in lane["ops"]:
            s, e = op_bounds(op)
            boundaries.add(s)
            boundaries.add(e)
    boundaries = sorted(b for b in boundaries if 0 <= b <= span)
    bidx = {b: i for i, b in enumerate(boundaries)}
    runs = [(boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)]

    def run_label(c0, c1):  # c1 is exclusive
        last = c1 - 1
        return str(c0) if last <= c0 else "{},...,{}".format(c0, last)

    labels = [run_label(b0, b1) for (b0, b1) in runs]

    # Variable column widths so abbreviated labels fit.
    col_w = [max(CELL_WIDTH, int(len(lbl) * CHAR_W) + 10) for lbl in labels]
    run_x = [LEFT_MARGIN]
    for w in col_w:
        run_x.append(run_x[-1] + w)
    chart_right = run_x[-1]
    chart_width = chart_right - LEFT_MARGIN

    # Reserve a right-side gutter for the per-resource utilization annotation.
    UTIL_GAP = 12
    util_texts = [util_label(lane["util"]) for lane in lanes if lane.get("util")]
    util_w = (int(max(len(t) for t in util_texts) * CHAR_W) + UTIL_GAP
              if util_texts else 0)
    util_x = chart_right + UTIL_GAP

    # Keep the canvas wide enough for the title so it is not clipped on short
    # charts (~9 px per char at font-size 16 bold, starting at LEFT_MARGIN).
    title = "Timetable: epoch {} (total_latency = {})".format(epoch, total)
    title_right = LEFT_MARGIN + int(len(title) * 9)
    width = max(chart_right + util_w, title_right) + RIGHT_MARGIN
    height = TOP_MARGIN + total_rows * LANE_HEIGHT + BOTTOM_MARGIN

    out = []
    out.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}" font-family="sans-serif">'.format(
            width, height, width, height
        )
    )
    out.append('<rect width="{}" height="{}" fill="white"/>'.format(width, height))

    # Title.
    out.append(
        '<text x="{}" y="22" font-size="16" font-weight="bold">{}</text>'.format(
            LEFT_MARGIN, esc(title)
        )
    )

    chart_bottom = TOP_MARGIN + total_rows * LANE_HEIGHT

    # Lane group backgrounds + separators, drawn first so gridlines and bars
    # sit on top. Each group is one lane and is as tall as its sub-row count.
    group_top_of = []
    row = 0
    for li, lane in enumerate(lanes):
        _, ntracks = lane_tracks[li]
        group_top = TOP_MARGIN + row * LANE_HEIGHT
        group_top_of.append(group_top)
        if li % 2 == 1:
            out.append(
                '<rect x="{}" y="{}" width="{}" height="{}" '
                'fill="#f7f7f7"/>'.format(
                    LEFT_MARGIN, group_top, chart_width, ntracks * LANE_HEIGHT
                )
            )
        if li > 0:
            out.append(
                '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#bbbbbb" '
                'stroke-width="1"/>'.format(
                    LEFT_MARGIN, group_top, chart_right, group_top
                )
            )
        row += ntracks

    # Column boundaries (gridlines) and the (possibly abbreviated) cycle range
    # centered above each column. A column is the half-open range [b_i, b_{i+1});
    # end is exclusive, so the column starting at an op's end is not covered by
    # that op.
    for i in range(len(runs) + 1):
        x = run_x[i]
        out.append(
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#dddddd" '
            'stroke-width="1"/>'.format(x, TOP_MARGIN, x, chart_bottom)
        )
    for i in range(len(runs)):
        xc = (run_x[i] + run_x[i + 1]) / 2
        out.append(
            '<text x="{}" y="{}" font-size="11" fill="#555" '
            'text-anchor="middle">{}</text>'.format(
                xc, TOP_MARGIN - 6, esc(labels[i])
            )
        )

    # Lane labels + operation bars (overlapping ops land on separate rows).
    for li, lane in enumerate(lanes):
        track_of, ntracks = lane_tracks[li]
        group_top = group_top_of[li]
        group_mid = group_top + ntracks * LANE_HEIGHT / 2 + 4
        out.append(
            '<text x="{}" y="{}" font-size="12" text-anchor="end">{}</text>'.format(
                LEFT_MARGIN - 10, group_mid, esc(lane["label"])
            )
        )
        # Per-resource utilization annotation in the right gutter.
        if lane.get("util"):
            out.append(
                '<text x="{}" y="{}" font-size="10" fill="#555" '
                'text-anchor="start">{}</text>'.format(
                    util_x, group_mid, esc(util_label(lane["util"]))
                )
            )

        ops = lane["ops"]
        for oi, op in enumerate(ops):
            s, e = op_bounds(op)
            i0 = bidx.get(s)
            i1 = bidx.get(e)
            if i0 is None or i1 is None:
                continue
            lane_top = group_top + track_of[oi] * LANE_HEIGHT
            x = run_x[i0]
            w = max(run_x[i1] - x, MIN_BAR_WIDTH)
            y = lane_top + BAR_PAD
            h = LANE_HEIGHT - 2 * BAR_PAD
            name = op["label"]
            if op.get("target"):
                # Control instruction echoed onto the slot it targets.
                tip = "{} (ctrl target) | port {} | start {} end {}".format(
                    name, op.get("port", ""), s, e
                )
            elif "duration" in op:
                tip = "{} | port {} | start {} end {} dur {}".format(
                    name, op.get("port", ""), s, e, op["duration"]
                )
            else:
                tip = "{} | start {} end {}".format(name, s, e)
            # Target echoes use a dashed outline so they read as control markers
            # rather than the datapath operation occupying the slot.
            dash = ' stroke-dasharray="2,1.5"' if op.get("target") else ""
            out.append(
                '<g><title>{}</title>'
                '<rect x="{}" y="{}" width="{}" height="{}" rx="3" '
                'fill="{}" fill-opacity="0.85" stroke="#333" '
                'stroke-width="0.7"{}/>'.format(
                    esc(tip), x, y, w, h, color_for(name), dash
                )
            )
            # Only draw the label when the bar is wide enough to hold it.
            if w >= 7 * len(str(name)) * 0.6:
                out.append(
                    '<text x="{}" y="{}" font-size="11" fill="white" '
                    'text-anchor="middle">{}</text>'.format(
                        x + w / 2, y + h / 2 + 4, esc(name)
                    )
                )
            out.append("</g>")

    out.append("</svg>")
    return "\n".join(out)


def svg_to_png(svg_path, png_path):
    """Rasterize an SVG to PNG using whatever system converter is available.
    Returns True on success, False if no converter could produce the PNG."""
    converters = [
        ["rsvg-convert", svg_path, "-o", png_path],
        ["inkscape", svg_path, "--export-type=png",
         "--export-filename=" + png_path],
        ["convert", svg_path, png_path],
        ["magick", svg_path, png_path],
    ]
    for cmd in converters:
        if not shutil.which(cmd[0]):
            continue
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            continue
    return False


def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
        else:
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Render schedule timetable JSON files as SVG Gantt charts."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="timetable JSON file(s) or directory/ies containing them",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="directory for the SVG output (default: alongside each input)",
    )
    args = parser.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("no timetable JSON files found")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for path in files:
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            print("skipping {}: {}".format(path, exc), file=sys.stderr)
            continue

        out_dir = args.output_dir or os.path.dirname(path) or "."
        epoch = data.get("epoch", "epoch")

        # Per-resource utilization (cycle accounting) as a sibling JSON file.
        utilization = compute_utilization(data)
        util_doc = {
            "epoch": epoch,
            "total_latency": int(data.get("total_latency", 0) or 0),
            "resources": utilization,
        }
        util_path = os.path.join(out_dir, "utilization_{}.json".format(epoch))
        with open(util_path, "w") as fh:
            json.dump(util_doc, fh, indent=2)
        print("wrote {}".format(util_path))

        svg = render(data, utilization)
        base = os.path.splitext(os.path.basename(path))[0]
        svg_path = os.path.join(out_dir, base + ".svg")
        with open(svg_path, "w") as fh:
            fh.write(svg)
        print("wrote {}".format(svg_path))

        png_path = os.path.join(out_dir, base + ".png")
        if svg_to_png(svg_path, png_path):
            print("wrote {}".format(png_path))
        else:
            print(
                "warning: no SVG->PNG converter found (rsvg-convert/inkscape/"
                "ImageMagick); kept SVG only",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
