// Render a fabric utilization heatmap as an SVG.
//
// This mirrors the spatial layout of `vs-component`'s `arch_visual_gen` (cells
// on a (row, col) grid, per-cell controller on the left and resource slots
// stacked on the right), but recolors each resource slot by its utilization
// instead of by resource kind. The geometry is intentionally kept close to the
// architecture view so the two read as the same fabric.
//
// Two JSON documents drive it:
//   * the resolved architecture JSON (the one `vs-component` writes and
//     `arch_visual_gen` consumes): a flat `cells` array of
//     `{ coordinates: {row, col}, cell: { controller, resources_list } }`,
//     where each resource carries `name`, `slot` and `size`.
//   * a `utilization_<epoch>.json` produced by the schedule stage
//     (`scripts/timetable.py`): `{ epoch, total_latency, resources: [ {row,
//     col, slot, acting_pct, config_pct, active_pct, stall_pct, ...} ] }`.
//
// A slot present in the architecture but absent from the utilization document
// (it ran no operations) is drawn in a distinct "no data" shade so the whole
// fabric is always shown.

use std::collections::HashMap;

use serde_json::Value;
use svg::node::element::{Group, Line, Rectangle, Text};
use svg::Document;

// Which utilization figure drives the slot color.
#[derive(Clone, Copy)]
pub enum Metric {
    Active,
    Acting,
    Config,
    Stall,
}

impl Metric {
    // Parse the CLI value; an unknown value returns None so the caller can
    // report it.
    pub fn from_arg(s: &str) -> Option<Metric> {
        match s.to_ascii_lowercase().as_str() {
            "active" => Some(Metric::Active),
            "acting" => Some(Metric::Acting),
            "config" => Some(Metric::Config),
            "stall" => Some(Metric::Stall),
            _ => None,
        }
    }

    // The percentage field to read from each utilization resource entry.
    fn pct_key(self) -> &'static str {
        match self {
            Metric::Active => "active_pct",
            Metric::Acting => "acting_pct",
            Metric::Config => "config_pct",
            Metric::Stall => "stall_pct",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Metric::Active => "active",
            Metric::Acting => "acting",
            Metric::Config => "config",
            Metric::Stall => "stall",
        }
    }
}

// Geometry constants, kept in step with arch_visual_gen so the heatmap and the
// architecture view line up.
const OFFSET: i64 = 5;
const CELL_WIDTH: i64 = 400;
const CELL_HEIGHT: i64 = 400;
const CONTROLLER_WIDTH: i64 = 80;
const CONTROLLER_HEIGHT: i64 = 320;
const RESOURCE_WIDTH: i64 = 260;
const RESOURCE_HEIGHT: i64 = 20;
const TITLE_H: i64 = 56;
const MARGIN: i64 = 12;
const LEGEND_W: i64 = 120;

// The YlOrRd sequential ramp (ColorBrewer): pale yellow (low) to deep red
// (high). Utilization reads intuitively as "hotter = busier".
const RAMP: [(f64, (i64, i64, i64)); 8] = [
    (0.0, (255, 255, 204)),
    (0.143, (255, 237, 160)),
    (0.286, (254, 217, 118)),
    (0.429, (254, 178, 76)),
    (0.571, (253, 141, 60)),
    (0.714, (252, 78, 42)),
    (0.857, (227, 26, 28)),
    (1.0, (177, 0, 38)),
];

// Interpolate the ramp at t in [0, 1], returning an "#rrggbb" string.
fn ramp_color(t: f64) -> String {
    let t = t.clamp(0.0, 1.0);
    for pair in RAMP.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t <= t1 {
            let span = t1 - t0;
            let f = if span > 0.0 { (t - t0) / span } else { 0.0 };
            let r = c0.0 as f64 + (c1.0 - c0.0) as f64 * f;
            let g = c0.1 as f64 + (c1.1 - c0.1) as f64 * f;
            let b = c0.2 as f64 + (c1.2 - c0.2) as f64 * f;
            return format!("#{:02x}{:02x}{:02x}", r as i64, g as i64, b as i64);
        }
    }
    "#b10026".to_string()
}

// Choose a readable text color for a filled slot: white on the darker (higher)
// end of the ramp, near-black on the paler low end.
fn text_on(pct: f64) -> &'static str {
    if pct >= 55.0 {
        "#ffffff"
    } else {
        "#222222"
    }
}

// Fill/border/text for a slot with no utilization data (never scheduled).
const NO_DATA_FILL: &str = "#e8e8e8";
const NO_DATA_BORDER: &str = "#b5b5b5";
const NO_DATA_TEXT: &str = "#777777";

// Build the (row, col, slot) -> percentage lookup for the chosen metric.
fn utilization_map(util: &Value, metric: Metric) -> HashMap<(i64, i64, i64), f64> {
    let mut map = HashMap::new();
    if let Some(resources) = util.get("resources").and_then(Value::as_array) {
        for r in resources {
            let row = r.get("row").and_then(Value::as_i64);
            let col = r.get("col").and_then(Value::as_i64);
            let slot = r.get("slot").and_then(Value::as_i64);
            let pct = r.get(metric.pct_key()).and_then(Value::as_f64);
            if let (Some(row), Some(col), Some(slot), Some(pct)) = (row, col, slot, pct) {
                map.insert((row, col, slot), pct);
            }
        }
    }
    map
}

// Render the heatmap document to an SVG string.
pub fn render(arch: &Value, util: &Value, metric: Metric) -> String {
    let util_map = utilization_map(util, metric);

    let cells = arch.get("cells").and_then(Value::as_array);
    // Grid extent from the cell coordinates (defensive against a missing/empty
    // architecture: fall back to a single cell so the canvas stays valid).
    let mut rows = 1;
    let mut cols = 1;
    if let Some(cells) = cells {
        for c in cells {
            let rr = c.pointer("/coordinates/row").and_then(Value::as_i64).unwrap_or(0);
            let cc = c.pointer("/coordinates/col").and_then(Value::as_i64).unwrap_or(0);
            rows = rows.max(rr + 1);
            cols = cols.max(cc + 1);
        }
    }

    let grid_w = cols * CELL_WIDTH;
    let grid_h = rows * CELL_HEIGHT;
    let width = MARGIN + grid_w + LEGEND_W + MARGIN;
    let height = TITLE_H + grid_h + MARGIN;

    let mut document = Document::new()
        .set("width", width)
        .set("height", height)
        .set("viewBox", (0, 0, width, height));
    document = document.add(
        Rectangle::new()
            .set("x", 0)
            .set("y", 0)
            .set("width", width)
            .set("height", height)
            .set("fill", "#ffffff"),
    );

    // Title.
    let epoch = util.get("epoch").and_then(Value::as_str).unwrap_or("epoch");
    let total = util.get("total_latency").and_then(Value::as_i64).unwrap_or(0);
    let title = format!(
        "Fabric utilization heatmap  \u{2014}  epoch {}  \u{2014}  {} %  (total_latency = {})",
        epoch,
        metric.label(),
        total
    );
    document = document.add(
        Text::new(title)
            .set("x", MARGIN)
            .set("y", 34)
            .set("fill", "#222222")
            .set("font-size", 20)
            .set("font-weight", "bold")
            .set("font-family", "sans-serif"),
    );

    if let Some(cells) = cells {
        for c in cells {
            let rr = c.pointer("/coordinates/row").and_then(Value::as_i64).unwrap_or(0);
            let cc = c.pointer("/coordinates/col").and_then(Value::as_i64).unwrap_or(0);
            let cell = match c.get("cell").and_then(Value::as_object) {
                Some(cell) => cell,
                None => {
                    continue;
                }
            };
            let x = MARGIN + cc * CELL_WIDTH;
            let y = TITLE_H + rr * CELL_HEIGHT;
            document = document.add(draw_cell(rr, cc, x, y, cell, &util_map, metric));
        }
    }

    document = document.add(draw_legend(MARGIN + grid_w + MARGIN, TITLE_H, metric));

    document.to_string()
}

// Draw one cell: its frame + coordinate label, a neutral controller block on
// the left, and the resource slots stacked on the right, each tinted by its
// utilization.
fn draw_cell(
    rr: i64,
    cc: i64,
    x: i64,
    y: i64,
    cell: &serde_json::Map<String, Value>,
    util_map: &HashMap<(i64, i64, i64), f64>,
    metric: Metric,
) -> Group {
    let mut group = Group::new();

    // Cell frame.
    group = group.add(
        Rectangle::new()
            .set("x", x + OFFSET)
            .set("y", y + OFFSET)
            .set("width", CELL_WIDTH - 2 * OFFSET)
            .set("height", CELL_HEIGHT - 2 * OFFSET)
            .set("fill", "#f4f4f4")
            .set("stroke", "#a5a5a5")
            .set("stroke-width", 2),
    );
    group = group.add(
        Text::new(format!("[{},{}]", rr, cc))
            .set("x", x + CELL_WIDTH / 2)
            .set("y", y + 40)
            .set("fill", "#323232")
            .set("font-size", 20)
            .set("font-weight", "bold")
            .set("text-anchor", "middle")
            .set("font-family", "sans-serif"),
    );

    // Controller block (neutral: utilization is reported per resource slot, not
    // for the controller itself).
    let ctrl_x = x + OFFSET + 21;
    let ctrl_y = y + OFFSET + 61;
    group = group.add(
        Rectangle::new()
            .set("x", ctrl_x)
            .set("y", ctrl_y)
            .set("width", CONTROLLER_WIDTH - 2)
            .set("height", CONTROLLER_HEIGHT - 2)
            .set("fill", "#fff0b0")
            .set("stroke", "#a65900")
            .set("stroke-width", 2),
    );
    if let Some(name) = cell.get("controller").and_then(|c| c.get("name")).and_then(Value::as_str) {
        let text_x = x + 11 + CONTROLLER_WIDTH / 2;
        let text_y = y + 11 + CONTROLLER_HEIGHT / 2;
        group = group.add(
            Text::new(name.to_string())
                .set("x", text_x)
                .set("y", text_y)
                .set("fill", "#a65900")
                .set("font-size", 18)
                .set("font-weight", "bold")
                .set("text-anchor", "middle")
                .set("font-family", "sans-serif")
                .set("transform", format!("rotate(90, {}, {})", text_x, text_y)),
        );
    }

    // Resource slots, stacked from the bottom upward (slot 0 at the bottom),
    // matching arch_visual_gen. Each is colored by its utilization.
    let resources = cell.get("resources_list").and_then(Value::as_array);
    let resource_x = x + OFFSET + 110;
    let mut resource_y = y + OFFSET + 50 + 16 * RESOURCE_HEIGHT;
    // Fallback slot index for resources that do not carry an explicit "slot".
    let mut running_slot = 0;
    if let Some(resources) = resources {
        for rs in resources {
            let resource = match rs.as_object() {
                Some(resource) => resource,
                None => {
                    continue;
                }
            };
            let size = resource.get("size").and_then(Value::as_i64).unwrap_or(1).max(1);
            let slot = resource.get("slot").and_then(Value::as_i64).unwrap_or(running_slot);
            let name = resource.get("name").and_then(Value::as_str).unwrap_or("");

            let pct = util_map.get(&(rr, cc, slot)).copied();
            let (fill, border, text_color) = match pct {
                Some(p) => (ramp_color(p / 100.0), "#333333".to_string(), text_on(p)),
                None => (
                    NO_DATA_FILL.to_string(),
                    NO_DATA_BORDER.to_string(),
                    NO_DATA_TEXT,
                ),
            };

            let box_x = resource_x + 11;
            let box_y = resource_y - RESOURCE_HEIGHT * size + 1;
            let box_w = RESOURCE_WIDTH - 12;
            let box_h = RESOURCE_HEIGHT * size - 2;

            // Tooltip carries the full breakdown for the slot.
            let tip = match pct {
                Some(p) => format!("{} | slot {} | {} {}%", name, slot, metric.label(), p as i64),
                None => format!("{} | slot {} | no schedule data", name, slot),
            };
            group = group.add(
                Rectangle::new()
                    .set("x", box_x)
                    .set("y", box_y)
                    .set("width", box_w)
                    .set("height", box_h)
                    .set("fill", fill)
                    .set("stroke", border)
                    .set("stroke-width", 2)
                    .add(svg::node::element::Title::new(tip)),
            );

            // Dashed separators for a multi-slot resource.
            if size > 1 {
                for i in 1..size {
                    let line_y = resource_y - RESOURCE_HEIGHT * (size - i) + 1;
                    group = group.add(
                        Line::new()
                            .set("x1", box_x)
                            .set("y1", line_y)
                            .set("x2", box_x + box_w)
                            .set("y2", line_y)
                            .set("stroke", "#555555")
                            .set("stroke-width", 1)
                            .set("stroke-dasharray", "5 5"),
                    );
                }
            }

            // Label: resource name and the utilization percentage.
            let label = match pct {
                Some(p) => format!("{}  {}%", name, p as i64),
                None => format!("{}  \u{2014}", name),
            };
            let text_x = box_x + box_w / 2;
            let text_y = box_y + box_h / 2 + 5;
            group = group.add(
                Text::new(label)
                    .set("x", text_x)
                    .set("y", text_y)
                    .set("fill", text_color)
                    .set("font-size", 15)
                    .set("font-weight", "bold")
                    .set("text-anchor", "middle")
                    .set("font-family", "sans-serif"),
            );

            resource_y -= RESOURCE_HEIGHT * size;
            running_slot = slot + size;
        }
    }

    group
}

// Draw the color legend (a vertical ramp from 100% at the top to 0% at the
// bottom) plus a "no data" swatch, at (x, y_top).
fn draw_legend(x: i64, y_top: i64, metric: Metric) -> Group {
    let mut group = Group::new();
    let bar_w = 26;
    let bar_h = 260;
    let bar_x = x;
    let bar_y = y_top + 30;
    let steps = 40;

    group = group.add(
        Text::new(format!("{} %", metric.label()))
            .set("x", bar_x)
            .set("y", bar_y - 10)
            .set("fill", "#222222")
            .set("font-size", 14)
            .set("font-weight", "bold")
            .set("font-family", "sans-serif"),
    );

    // Stacked bands, high value on top.
    for i in 0..steps {
        let t = 1.0 - (i as f64 + 0.5) / steps as f64;
        let seg_y = bar_y + i * bar_h / steps;
        let seg_h = bar_h / steps + 1;
        group = group.add(
            Rectangle::new()
                .set("x", bar_x)
                .set("y", seg_y)
                .set("width", bar_w)
                .set("height", seg_h)
                .set("fill", ramp_color(t)),
        );
    }
    group = group.add(
        Rectangle::new()
            .set("x", bar_x)
            .set("y", bar_y)
            .set("width", bar_w)
            .set("height", bar_h)
            .set("fill", "none")
            .set("stroke", "#333333")
            .set("stroke-width", 1),
    );

    // Tick labels at 0/25/50/75/100.
    for pct in [0, 25, 50, 75, 100] {
        let tick_y = bar_y + ((100 - pct) * bar_h / 100) + 5;
        group = group.add(
            Text::new(format!("{}", pct))
                .set("x", bar_x + bar_w + 6)
                .set("y", tick_y)
                .set("fill", "#333333")
                .set("font-size", 12)
                .set("font-family", "sans-serif"),
        );
    }

    // "No data" swatch below the bar.
    let nd_y = bar_y + bar_h + 24;
    group = group.add(
        Rectangle::new()
            .set("x", bar_x)
            .set("y", nd_y)
            .set("width", bar_w)
            .set("height", 18)
            .set("fill", NO_DATA_FILL)
            .set("stroke", NO_DATA_BORDER)
            .set("stroke-width", 1),
    );
    group = group.add(
        Text::new("no data")
            .set("x", bar_x + bar_w + 6)
            .set("y", nd_y + 14)
            .set("fill", "#333333")
            .set("font-size", 12)
            .set("font-family", "sans-serif"),
    );

    group
}
