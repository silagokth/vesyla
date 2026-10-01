use clap::{error::ErrorKind, Parser, Subcommand};
use dialoguer::{theme::ColorfulTheme, Select};
use log::{error, info, warn, LevelFilter};
use serde_json::Value;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{self, Stdio};

mod heatmap;

#[derive(Subcommand)]
enum Command {
    #[command(
        about = "Show the schedule timetable(s) produced during compilation",
        name = "timetable"
    )]
    Timetable {
        /// Compile output directory (the one passed as --output to `vesyla compile`)
        #[arg(short, long, default_value = ".")]
        directory: String,
    },
    #[command(
        about = "Show the interconnect routing graph(s) produced during compilation",
        name = "interconnect"
    )]
    Interconnect {
        /// Compile output directory (the one passed as --output to `vesyla compile`)
        #[arg(short, long, default_value = ".")]
        directory: String,
    },
    #[command(
        about = "Show the constraint graph(s) produced during compilation",
        name = "constraint"
    )]
    Constraint {
        /// Compile output directory (the one passed as --output to `vesyla compile`)
        #[arg(short, long, default_value = ".")]
        directory: String,
    },
    #[command(
        about = "Render and show the resource conflict graph(s) produced during compilation",
        name = "conflict-graph"
    )]
    ConflictGraph {
        /// Compile output directory (the one passed as --output to `vesyla compile`)
        #[arg(short, long, default_value = ".")]
        directory: String,
    },
    #[command(
        about = "Render and show a fabric utilization heatmap produced during compilation",
        name = "heatmap"
    )]
    Heatmap {
        /// Compile output directory (the one passed as --output to `vesyla compile`)
        #[arg(short, long, default_value = ".")]
        directory: String,
        /// Resolved architecture JSON (the arch.json emitted by `vs-component`,
        /// with per-cell coordinates) that supplies the fabric geometry. If
        /// omitted (or if an input-form arch.json is given), the resolved
        /// arch.json for the run is auto-discovered under --directory.
        #[arg(short, long)]
        arch: Option<String>,
        /// Utilization figure to color slots by: active, acting, config, or stall
        #[arg(short, long, default_value = "active")]
        metric: String,
    },
    #[command(
        about = "Show the fabric architecture diagram (fabric.svg) generated for the design",
        name = "fabric"
    )]
    Fabric {
        /// Directory to search for fabric.svg/.png (the compile/assembly output directory)
        #[arg(short, long, default_value = ".")]
        directory: String,
    },
    #[command(
        about = "Print the assembly instructions (instr.asm) produced during compilation",
        name = "instructions"
    )]
    Instructions {
        /// Compile output directory (the one passed as --output to `vesyla compile`)
        #[arg(short, long, default_value = ".")]
        directory: String,
    },
    #[command(
        about = "Open the simulation waveform (trace.vcd) in gtkwave",
        name = "wave"
    )]
    Wave {
        /// Directory to search for trace.vcd (the simulation output directory)
        #[arg(short, long, default_value = ".")]
        directory: String,
        /// gtkwave save file (.gtkw) with the signals/layout to preload. If
        /// omitted, $VESYLA_WAVE_SAVE or a .gtkw next to the trace is used.
        #[arg(short, long)]
        save: Option<String>,
    },
}

#[derive(Parser)]
#[command(about, long_about = None, allow_missing_positional = true, after_help = "")]
struct Args {
    /// Debug artifact to show
    #[command(subcommand)]
    command: Command,
}

fn main() -> Result<(), io::Error> {
    // set logger level to be debug
    env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .init();
    log_panics::init();

    // make sure the program returns a non-zero status code when arguments are invalid
    let cli_args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => {
            match e.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                    println!("{}", e);
                    return Ok(());
                }
                _ => {
                    error!("{}", e);
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Invalid arguments: {}", e),
                    ));
                }
            }
        }
    };

    match &cli_args.command {
        Command::Timetable { directory } => {
            show_artifact(directory, "timetable_dir", "debug/timetable", "timetable")
        }
        Command::Interconnect { directory } => show_artifact(
            directory,
            "interconnect_dir",
            "debug/interconnect",
            "interconnect graph",
        ),
        Command::Constraint { directory } => show_artifact(
            directory,
            "constraint_dir",
            "debug/constraint",
            "constraint graph",
        ),
        Command::ConflictGraph { directory } => show_conflict_graph(directory),
        Command::Heatmap {
            directory,
            arch,
            metric,
        } => show_heatmap(directory, arch.as_deref(), metric),
        Command::Fabric { directory } => show_fabric(directory),
        Command::Instructions { directory } => show_instructions(directory),
        Command::Wave { directory, save } => show_wave(directory, save),
    }
}

// Locate the runtime config.json the same way vs-compile does: it sits in
// "config/config.json" relative to the program directory, which is the parent
// of the executable's directory (e.g. "<root>/bin/vs-show" -> "<root>/").
fn prog_dir() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let dir = exe.parent()?.parent()?;
    fs::canonicalize(dir).ok()
}

// Substitute "${key}" references in `value` with other entries of the same
// section. Mirrors vesyla::pasm::Config::resolve (Config.cpp) including the
// depth bound that guards against a cyclic reference in a hand-edited config.
fn resolve(section: &serde_json::Map<String, Value>, value: &str, depth: u32) -> String {
    if depth == 0 {
        return value.to_string();
    }
    let mut result = value.to_string();
    let mut pos = 0;
    while let Some(rel) = result[pos..].find("${") {
        let start = pos + rel;
        let end = match result[start + 2..].find('}') {
            Some(e) => start + 2 + e,
            None => {
                break;
            }
        };
        let key = result[start + 2..end].to_string();
        let raw = section
            .get(&key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let replacement = resolve(section, &raw, depth - 1);
        result.replace_range(start..=end, &replacement);
        pos = start + replacement.len();
    }
    result
}

// Read a resolved value from the "output" section of the runtime config.json,
// falling back to `default` if the config or key cannot be found.
fn output_config(key: &str, default: &str) -> String {
    let config_file = match prog_dir() {
        Some(dir) => dir.join("config/config.json"),
        None => {
            return default.to_string();
        }
    };
    let content = match fs::read_to_string(&config_file) {
        Ok(c) => c,
        Err(_) => {
            warn!(
                "Could not read config {:?}, using default {}={}",
                config_file, key, default
            );
            return default.to_string();
        }
    };
    let json: Value = match serde_json::from_str(&content) {
        Ok(j) => j,
        Err(e) => {
            warn!("Could not parse config {:?}: {}", config_file, e);
            return default.to_string();
        }
    };
    let section = match json.get("output").and_then(Value::as_object) {
        Some(s) => s,
        None => {
            return default.to_string();
        }
    };
    match section.get(key).and_then(Value::as_str) {
        Some(raw) => resolve(section, raw, 16),
        None => default.to_string(),
    }
}

// Discover the debug images for one artifact kind (timetable, interconnect, ...)
// and let the user open one. config.json only defines the artifact directory
// *suffix* (e.g. "debug/timetable"); the base output directory is chosen by
// whoever runs `vesyla compile` (the testcase scripts move it to
// work/archive/compile_<id>). So we take the configured suffix and recursively
// discover every matching directory under `directory`.
fn show_artifact(
    directory: &str,
    config_key: &str,
    default_suffix: &str,
    name: &str,
) -> Result<(), io::Error> {
    let dir_suffix = output_config(config_key, default_suffix);
    let root = Path::new(directory);

    let mut images: Vec<PathBuf> = Vec::new();
    collect_images(root, Path::new(&dir_suffix), &mut images)?;

    // Each graph is rendered as both a ".svg" and a ".png"; prefer the (higher-
    // quality vector) SVG and only fall back to the PNG when no SVG exists.
    // Dedupe by the extension-less path so we show one entry per graph.
    let images = prefer_svg(images);

    if images.is_empty() {
        error!(
            "No {} images found under {:?} (looked for '{}' directories). \
             Did you run `vesyla compile` first?",
            name, root, dir_suffix
        );
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("No {} images found under {:?}", name, root),
        ));
    }

    // Present each image (path relative to the search root, so graphs from
    // different compile runs are distinguishable) and let the user pick one.
    let labels: Vec<String> = images
        .iter()
        .map(|p| {
            p.strip_prefix(root)
                .unwrap_or(p)
                .to_str()
                .unwrap_or("<unknown>")
                .to_string()
        })
        .collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt(format!(
            "Select a {} to open (↑/↓ to move, Enter to open, Esc to cancel)",
            name
        ))
        .items(&labels)
        .default(0)
        .interact_opt()
        .map_err(|e| io::Error::other(format!("Selection failed: {}", e)))?;

    let index = match selection {
        Some(i) => i,
        None => {
            info!("Nothing selected.");
            return Ok(());
        }
    };

    open_in_viewer(&images[index])
}

// Find the conflict-graph .dot file(s) produced during compilation, render the
// chosen one to SVG with graphviz `dot`, and open it. Unlike the interconnect/
// constraint graphs (which are rendered to images at compile time), the conflict
// graph is emitted only as .dot, so it is rendered here on demand. The directory
// suffix comes from config.json ("conflict_graph_dir").
fn show_conflict_graph(directory: &str) -> Result<(), io::Error> {
    let dir_suffix = output_config("conflict_graph_dir", "debug/conflict_graph");
    let root = Path::new(directory);

    let mut dots: Vec<PathBuf> = Vec::new();
    collect_dots(root, Path::new(&dir_suffix), &mut dots)?;
    dots.sort();

    if dots.is_empty() {
        error!(
            "No conflict graph .dot files found under {:?} (looked for '{}' \
             directories). Did you run `vesyla compile` first?",
            root, dir_suffix
        );
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("No conflict graph .dot files found under {:?}", root),
        ));
    }

    let dot = match pick_one(
        &dots,
        root,
        "Select a conflict graph to open (↑/↓ to move, Enter to open, Esc to cancel)",
    )? {
        Some(f) => f,
        None => {
            info!("Nothing selected.");
            return Ok(());
        }
    };

    let svg = render_dot(dot)?;
    open_in_viewer(&svg)
}

// Render a fabric utilization heatmap for a chosen epoch and open it. The
// utilization_<epoch>.json files are written next to the schedule timetables
// (the "timetable_dir" configured for `vesyla compile`); the fabric geometry
// comes from the resolved architecture JSON emitted by `vs-component`, passed
// via --arch. One heatmap is produced per epoch (the selected utilization file).
fn show_heatmap(directory: &str, arch: Option<&str>, metric: &str) -> Result<(), io::Error> {
    let metric = match heatmap::Metric::from_arg(metric) {
        Some(m) => m,
        None => {
            error!(
                "Unknown metric '{}'. Choose one of: active, acting, config, stall.",
                metric
            );
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unknown metric '{}'", metric),
            ));
        }
    };

    // Discover the per-epoch utilization files under the timetable directory.
    let dir_suffix = output_config("timetable_dir", "compile/timetable");
    let root = Path::new(directory);
    let mut files: Vec<PathBuf> = Vec::new();
    collect_utilization(root, Path::new(&dir_suffix), &mut files)?;
    files.sort();

    if files.is_empty() {
        error!(
            "No utilization_*.json files found under {:?} (looked for '{}' \
             directories). Did you run `vesyla compile` first?",
            root, dir_suffix
        );
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("No utilization files found under {:?}", root),
        ));
    }

    let util_file = match pick_one(
        &files,
        root,
        "Select an epoch's utilization to view (↑/↓ to move, Enter to open, Esc to cancel)",
    )? {
        Some(f) => f,
        None => {
            info!("Nothing selected.");
            return Ok(());
        }
    };

    let util_content = fs::read_to_string(util_file)?;
    let util_json: Value = serde_json::from_str(&util_content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // Resolve the fabric geometry: the *resolved* arch.json (with per-cell
    // coordinates) that `vs-component` emits. The input-form arch.json (cell
    // templates, no coordinates) has no placement and would draw nothing.
    let arch_json = load_geometry(arch, util_file, root)?;

    let svg = heatmap::render(&arch_json, &util_json, metric);

    // Write the heatmap SVG next to its utilization file, then open it.
    let epoch = util_json
        .get("epoch")
        .and_then(Value::as_str)
        .unwrap_or("epoch");
    let out_path = util_file
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("heatmap_{}.svg", epoch));
    fs::write(&out_path, svg)?;
    info!("Wrote fabric heatmap: {}", out_path.display());

    open_in_viewer(&out_path)
}

// Recursively walk `root`, collecting every "utilization_*.json" that lives in a
// directory whose path ends with `suffix` (the configured timetable directory,
// e.g. "compile/timetable"). file_type() is used instead of is_dir() so symlinks
// are not followed, which avoids cycles.
fn collect_utilization(
    root: &Path,
    suffix: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), io::Error> {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        // A directory we cannot read (permissions, races) should not abort the
        // whole search; just skip it.
        Err(_) => {
            return Ok(());
        }
    };

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();

        if path.ends_with(suffix) {
            for e in fs::read_dir(&path)? {
                let file = e?.path();
                let is_util = file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("utilization_") && n.ends_with(".json"))
                    .unwrap_or(false);
                if file.is_file() && is_util {
                    out.push(file);
                }
            }
        } else {
            collect_utilization(&path, suffix, out)?;
        }
    }
    Ok(())
}

// Resolve the fabric geometry document for the heatmap. An explicit --arch is
// used when it is already the resolved form; otherwise (omitted, or an input-
// form arch.json) the resolved arch.json for the run is auto-discovered.
fn load_geometry(
    arch: Option<&str>,
    util_file: &Path,
    root: &Path,
) -> Result<Value, io::Error> {
    if let Some(arch) = arch {
        let path = Path::new(arch);
        let content = fs::read_to_string(path).map_err(|e| {
            error!("Could not read architecture file {:?}: {}", path, e);
            e
        })?;
        let value: Value = serde_json::from_str(&content).map_err(|e| {
            error!("Could not parse architecture file {:?}: {}", path, e);
            io::Error::new(io::ErrorKind::InvalidData, e)
        })?;
        if is_resolved_arch(&value) {
            return Ok(value);
        }
        warn!(
            "{:?} is an input architecture description (its cells carry no \
             coordinates), not the resolved fabric. Searching for the resolved \
             arch.json produced by `vs-component`...",
            path
        );
    }

    match find_resolved_arch(util_file, root) {
        Some(found) => {
            info!("Using resolved architecture geometry: {}", found.display());
            let content = fs::read_to_string(&found)?;
            serde_json::from_str(&content)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        }
        None => {
            error!(
                "Could not find a resolved arch.json (with per-cell coordinates and \
                 resources_list) under {:?}. Pass it with --arch; it is the arch.json \
                 emitted by `vs-component` (e.g. work/system/arch/arch.json).",
                root
            );
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no resolved architecture geometry found",
            ))
        }
    }
}

// A resolved fabric has a top-level `cells` array whose entries carry a
// `coordinates` object and a nested `cell` (the input-form arch.json instead
// lists cell templates with a `resource_list` and no coordinates).
fn is_resolved_arch(v: &Value) -> bool {
    v.get("cells")
        .and_then(Value::as_array)
        .and_then(|cells| cells.first())
        .map(|c0| c0.get("coordinates").is_some() && c0.get("cell").is_some())
        .unwrap_or(false)
}

fn is_resolved_arch_file(path: &Path) -> bool {
    match fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str::<Value>(&content) {
            Ok(v) => is_resolved_arch(&v),
            Err(_) => false,
        },
        Err(_) => false,
    }
}

// Locate the resolved architecture JSON for the run that produced `util_file`.
// Preference: the arch.json in the same work-tree as the utilization file (the
// known `system/arch/arch.json` or `archive/assemble/arch/arch.json` above it);
// failing that, any resolved arch.json found under `root`.
fn find_resolved_arch(util_file: &Path, root: &Path) -> Option<PathBuf> {
    let mut dir = util_file.parent();
    while let Some(d) = dir {
        for rel in ["system/arch/arch.json", "archive/assemble/arch/arch.json"] {
            let candidate = d.join(rel);
            if is_resolved_arch_file(&candidate) {
                return Some(candidate);
            }
        }
        if d == root {
            break;
        }
        dir = d.parent();
    }

    // Fall back to any resolved arch.json under the search root.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if collect_named(root, "arch.json", &mut candidates).is_ok() {
        candidates.sort();
        for candidate in candidates {
            if is_resolved_arch_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

// Render a Graphviz .dot file to an SVG next to it using the `dot` binary,
// returning the generated SVG path. A missing `dot` yields a clear, actionable
// error instead of failing silently.
fn render_dot(dot: &Path) -> Result<PathBuf, io::Error> {
    let svg = dot.with_extension("svg");
    let result = process::Command::new("dot")
        .arg("-Tsvg")
        .arg(dot)
        .arg("-o")
        .arg(&svg)
        .status();

    match result {
        Ok(status) if status.success() => {
            info!("Rendered {} -> {}", dot.display(), svg.display());
            Ok(svg)
        }
        Ok(status) => {
            error!("`dot` failed to render {} ({})", dot.display(), status);
            Err(io::Error::other(format!(
                "dot failed to render {}",
                dot.display()
            )))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            error!(
                "`dot` is not available on this system. Please install it (e.g. the \
                 graphviz package) or render the file manually: dot -Tsvg {} -o {}",
                dot.display(),
                svg.display()
            );
            Err(io::Error::new(io::ErrorKind::NotFound, "dot is not available"))
        }
        Err(e) => {
            error!("Failed to run `dot` on {}: {}", dot.display(), e);
            Err(e)
        }
    }
}

// Recursively walk `root`, collecting every ".dot" that lives in a directory
// whose path ends with `suffix` (the configured conflict-graph directory, e.g.
// "debug/conflict_graph"). file_type() is used instead of is_dir() so symlinks
// are not followed, which avoids cycles.
fn collect_dots(root: &Path, suffix: &Path, out: &mut Vec<PathBuf>) -> Result<(), io::Error> {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        // A directory we cannot read (permissions, races) should not abort the
        // whole search; just skip it.
        Err(_) => {
            return Ok(());
        }
    };

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();

        if path.ends_with(suffix) {
            for e in fs::read_dir(&path)? {
                let file = e?.path();
                let is_dot = file
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("dot"))
                    .unwrap_or(false);
                if file.is_file() && is_dot {
                    out.push(file);
                }
            }
        } else {
            collect_dots(&path, suffix, out)?;
        }
    }
    Ok(())
}

// Find the fabric architecture diagram(s) (fabric.svg / fabric.png) generated
// by `vs-component` and open the chosen one. Unlike the debug graphs, the fabric
// diagram is a plain named file living in the assembly arch output (e.g.
// system/arch/fabric.svg), so it is discovered by name anywhere under
// `directory`. The SVG is preferred over a sibling PNG.
fn show_fabric(directory: &str) -> Result<(), io::Error> {
    let root = Path::new(directory);

    let mut images: Vec<PathBuf> = Vec::new();
    collect_named(root, "fabric.svg", &mut images)?;
    collect_named(root, "fabric.png", &mut images)?;
    let images = prefer_svg(images);

    if images.is_empty() {
        error!(
            "No fabric.svg/.png found under {:?}. Did you run `vesyla` (component \
             assembly) first?",
            root
        );
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("No fabric diagram found under {:?}", root),
        ));
    }

    let image = match pick_one(
        &images,
        root,
        "Select a fabric diagram to open (↑/↓ to move, Enter to open, Esc to cancel)",
    )? {
        Some(f) => f,
        None => {
            info!("Nothing selected.");
            return Ok(());
        }
    };

    open_in_viewer(image)
}

// Find the assembly instruction file(s) (instr.asm) produced by compilation and
// print the chosen one to stdout. Like the graph artifacts, instr.asm lives
// under a base output directory the testcase scripts move to
// work/archive/compile_<id>, so we recursively discover every match under
// `directory`. The basename comes from config.json ("instr_basename").
fn show_instructions(directory: &str) -> Result<(), io::Error> {
    let filename = format!("{}.asm", output_config("instr_basename", "instr"));
    let root = Path::new(directory);

    let mut files: Vec<PathBuf> = Vec::new();
    collect_named(root, &filename, &mut files)?;
    files.sort();

    if files.is_empty() {
        error!(
            "No {} files found under {:?}. Did you run `vesyla compile` first?",
            filename, root
        );
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("No {} files found under {:?}", filename, root),
        ));
    }

    let file = match pick_one(
        &files,
        root,
        "Select the instructions to print (↑/↓ to move, Enter to print, Esc to cancel)",
    )? {
        Some(f) => f,
        None => {
            info!("Nothing selected.");
            return Ok(());
        }
    };

    let content = fs::read_to_string(file)?;
    print_asm(&content);
    Ok(())
}

// Open the simulation waveform (trace.vcd) in gtkwave. Like the other artifacts,
// trace.vcd is written under a run directory (debug/trace.vcd) that may live in
// work/archive/compile_<id>, so we recursively discover every match. A gtkwave
// save file (.gtkw), if found, is passed alongside so the curated signals/layout
// are preloaded.
fn show_wave(directory: &str, save: &Option<String>) -> Result<(), io::Error> {
    let root = Path::new(directory);

    let mut files: Vec<PathBuf> = Vec::new();
    collect_named(root, "trace.vcd", &mut files)?;
    files.sort();

    if files.is_empty() {
        error!(
            "No trace.vcd files found under {:?}. Did you run the RTL simulation \
             (in debug mode) first?",
            root
        );
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("No trace.vcd files found under {:?}", root),
        ));
    }

    let file = match pick_one(
        &files,
        root,
        "Select a waveform to open (↑/↓ to move, Enter to open, Esc to cancel)",
    )? {
        Some(f) => f,
        None => {
            info!("Nothing selected.");
            return Ok(());
        }
    };

    // Resolve the save file: an existing one (--save / env / sibling) is reused
    // as-is; otherwise generate one from the trace's own signal hierarchy so the
    // default view matches whatever cells this design actually instantiated.
    let save_file = match resolve_save_file(save, file)? {
        Some(s) => {
            info!("Preloading gtkwave save file: {}", s.display());
            Some(s)
        }
        None => match generate_save_file(file) {
            Ok(Some(generated)) => {
                info!("Generated gtkwave save file: {}", generated.display());
                Some(generated)
            }
            Ok(None) => {
                warn!(
                    "No default signals matched in {}; opening without a save file.",
                    file.display()
                );
                None
            }
            Err(e) => {
                warn!("Could not generate gtkwave save file: {}", e);
                None
            }
        },
    };

    // gtkwave takes the dump file first, then an optional save file.
    let mut args: Vec<OsString> = vec![file.as_os_str().to_owned()];
    if let Some(ref s) = save_file {
        args.push(s.as_os_str().to_owned());
    }

    let arg_refs: Vec<&OsStr> = args.iter().map(OsString::as_os_str).collect();
    open_with("gtkwave", "the gtkwave package", &arg_refs)
}

// Generate a gtkwave save file (.gtkw) next to `vcd` from the trace's own signal
// hierarchy, so the default view adapts to whichever cells the design has. It
// selects the fabric data IO (fabric_inst.io_data_in/out) and every per-cell
// controller instruction register (cell_*_inst.controller_inst.instr).
// Returns Ok(None) if no matching signals were found.
fn generate_save_file(vcd: &Path) -> Result<Option<PathBuf>, io::Error> {
    use std::io::BufRead;

    // A resource is a scope inside a cell; we collect a fixed set of its data/
    // activation signals. The controller's instruction register is collected
    // separately per cell. Signal lists are keyed by the range-less name so
    // bit-blasted vectors (instr[31], instr[30], ...) coalesce into one bus.
    struct Resource {
        display: String,
        signals: Vec<(String, Width)>,
    }
    struct Cell {
        display: String,
        controller: Vec<(String, Width)>,
        resources: Vec<(String, Resource)>, // keyed by resource scope path
    }

    // The per-resource signals to preload, matched by their base name.
    const RESOURCE_SIGNALS: [&str; 9] = [
        "activate_0",
        "word_data_in_0",
        "word_data_in_1",
        "word_data_out_0",
        "word_data_out_1",
        "bulk_data_in_0",
        "bulk_data_in_1",
        "bulk_data_out_0",
        "bulk_data_out_1",
    ];

    let is_cell = |s: &str| s.starts_with("cell_") && s.ends_with("_inst");
    let strip_inst = |s: &str| s.strip_suffix("_inst").unwrap_or(s).to_string();

    let reader = io::BufReader::new(fs::File::open(vcd)?);
    let mut scopes: Vec<String> = Vec::new();
    let mut io_signals: Vec<(String, Width)> = Vec::new();
    let mut cells: Vec<(String, Cell)> = Vec::new(); // keyed by cell scope path

    // Only the VCD definition header carries the hierarchy; stop before the
    // (potentially huge) value-change section.
    for line in reader.lines() {
        let line = line?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("$scope") => {
                // $scope <type> <name> $end
                if tokens.len() >= 3 {
                    scopes.push(tokens[2].to_string());
                }
            }
            Some("$upscope") => {
                scopes.pop();
            }
            Some("$enddefinitions") => {
                break;
            }
            Some("$var") => {
                // $var <type> <size> <id> <ref> [range] $end — the range may be
                // its own token ("instr [31:0]") or attached ("instr[31:0]").
                if tokens.len() < 6 {
                    continue;
                }
                let (base, mut range) = split_range(tokens[4]);
                if range.is_empty() && tokens[5] != "$end" {
                    range = tokens[5].to_string();
                }
                let base_full = format!("{}.{}", scopes.join("."), base);
                let last = scopes.last().map(String::as_str).unwrap_or("");

                // Fabric-level data IO.
                if last == "fabric_inst" && (base == "io_data_in" || base == "io_data_out") {
                    add_signal(&mut io_signals, &base_full, &range);
                    continue;
                }

                // Everything else of interest lives in a scope beneath a cell.
                let cell_idx = match scopes.iter().position(|s| is_cell(s)) {
                    Some(i) => i,
                    None => continue,
                };
                // The signal must sit in a scope below the cell (the controller
                // or a resource), not directly in the cell scope itself.
                if scopes.len() - 1 == cell_idx {
                    continue;
                }

                let is_instr = last == "controller_inst" && base == "instr";
                let is_resource_signal = last != "controller_inst" && RESOURCE_SIGNALS.contains(&base);
                if !is_instr && !is_resource_signal {
                    continue;
                }

                // Get or create the owning cell.
                let cell_path = scopes[..=cell_idx].join(".");
                let cell_display = strip_inst(&scopes[cell_idx]);
                let ci = match cells.iter().position(|(k, _)| *k == cell_path) {
                    Some(i) => i,
                    None => {
                        cells.push((
                            cell_path,
                            Cell {
                                display: cell_display,
                                controller: Vec::new(),
                                resources: Vec::new(),
                            },
                        ));
                        cells.len() - 1
                    }
                };
                let cell = &mut cells[ci].1;

                if is_instr {
                    add_signal(&mut cell.controller, &base_full, &range);
                } else {
                    // Get or create the owning resource (keyed by its scope path).
                    let res_path = scopes.join(".");
                    let ri = match cell.resources.iter().position(|(k, _)| *k == res_path) {
                        Some(i) => i,
                        None => {
                            cell.resources.push((
                                res_path,
                                Resource {
                                    display: strip_inst(last),
                                    signals: Vec::new(),
                                },
                            ));
                            cell.resources.len() - 1
                        }
                    };
                    add_signal(&mut cell.resources[ri].1.signals, &base_full, &range);
                }
            }
            _ => {}
        }
    }

    if io_signals.is_empty() && cells.is_empty() {
        return Ok(None);
    }

    // Emit the save file. Comment rows (lines starting with '-') label each cell
    // and resource; an "@" flags line (hex radix) precedes each signal group.
    // gtkwave applies this to the dump given on the command line, so no
    // [dumpfile] header is needed.
    let mut out = String::from("[*] vesyla-generated default view\n[timestart] 0\n");

    if !io_signals.is_empty() {
        out.push_str("-fabric IO\n@28\n");
        emit_signals(&mut out, &mut io_signals);
    }

    for (_, cell) in &mut cells {
        out.push_str(&format!("-{}\n", cell.display));
        if !cell.controller.is_empty() {
            out.push_str("-  controller\n@28\n");
            emit_signals(&mut out, &mut cell.controller);
        }
        for (_, res) in &mut cell.resources {
            out.push_str(&format!("-  {}\n@28\n", res.display));
            emit_signals(&mut out, &mut res.signals);
        }
    }

    let path = vcd.with_extension("gtkw");
    fs::write(&path, out)?;
    Ok(Some(path))
}

// The declared width of a signal, used to render one collapsed reference. A
// bit-blasted vector (each bit declared as its own single-index $var) is tracked
// as Blasted and later rendered as a single "name[hi:lo]" bus.
enum Width {
    Scalar,
    Bus(String), // range token including brackets, e.g. "[31:0]"
    Blasted { hi: i64, lo: i64 },
}

// Record a signal (keyed by its range-less name) into `list`, merging repeated
// single-bit declarations of the same vector into one Blasted entry.
fn add_signal(list: &mut Vec<(String, Width)>, base_full: &str, range: &str) {
    let pos = list.iter().position(|(k, _)| k == base_full);
    let inner = range.trim_start_matches('[').trim_end_matches(']');

    // A single-bit slice like "[3]" (an index, no ':') is one bit of a
    // bit-blasted bus; accumulate its extent instead of listing each bit.
    if !range.is_empty() && !inner.contains(':') {
        if let Ok(idx) = inner.parse::<i64>() {
            match pos {
                Some(i) => {
                    if let Width::Blasted { hi, lo } = &mut list[i].1 {
                        *hi = (*hi).max(idx);
                        *lo = (*lo).min(idx);
                    }
                }
                None => list.push((base_full.to_string(), Width::Blasted { hi: idx, lo: idx })),
            }
            return;
        }
    }

    // Scalar (no range) or an already-collapsed bus ("[31:0]").
    let width = if range.is_empty() {
        Width::Scalar
    } else {
        Width::Bus(range.to_string())
    };
    match pos {
        Some(i) => list[i].1 = width,
        None => list.push((base_full.to_string(), width)),
    }
}

// Sort a signal list by name and append the rendered (collapsed) references.
fn emit_signals(out: &mut String, list: &mut [(String, Width)]) {
    list.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, width) in list.iter() {
        match width {
            Width::Scalar => out.push_str(name),
            Width::Bus(range) => {
                out.push_str(name);
                out.push_str(range);
            }
            Width::Blasted { hi, lo } => {
                out.push_str(&format!("{}[{}:{}]", name, hi, lo));
            }
        }
        out.push('\n');
    }
}

// Split a VCD reference into its base name and bit range: "instr[31:0]" ->
// ("instr", "[31:0]"); "clk" -> ("clk", "").
fn split_range(reference: &str) -> (&str, String) {
    match reference.find('[') {
        Some(i) => (&reference[..i], reference[i..].to_string()),
        None => (reference, String::new()),
    }
}

// Resolve the gtkwave save file (.gtkw) to preload, in priority order:
//   1. the explicit --save argument (an error if it does not exist);
//   2. $VESYLA_WAVE_SAVE (skipped with a warning if it does not exist);
//   3. auto-discovery next to the chosen trace.vcd — a sibling matching its stem
//      (e.g. trace.gtkw), else any *.gtkw in the same directory.
// Returns Ok(None) when no save file is configured or found.
fn resolve_save_file(save: &Option<String>, vcd: &Path) -> Result<Option<PathBuf>, io::Error> {
    if let Some(s) = save {
        let path = PathBuf::from(s);
        if !path.is_file() {
            error!("gtkwave save file not found: {:?}", path);
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("gtkwave save file not found: {:?}", path),
            ));
        }
        return Ok(Some(path));
    }

    if let Some(env_save) = env::var_os("VESYLA_WAVE_SAVE") {
        let path = PathBuf::from(env_save);
        if path.is_file() {
            return Ok(Some(path));
        }
        warn!(
            "$VESYLA_WAVE_SAVE points to a missing file ({:?}); ignoring it.",
            path
        );
    }

    Ok(find_sibling_save_file(vcd))
}

// Look for a gtkwave save file in the same directory as `vcd`: first a sibling
// sharing the dump's stem (trace.vcd -> trace.gtkw), then any other *.gtkw
// (sorted, first wins).
fn find_sibling_save_file(vcd: &Path) -> Option<PathBuf> {
    let preferred = vcd.with_extension("gtkw");
    if preferred.is_file() {
        return Some(preferred);
    }

    let dir = vcd.parent()?;
    let mut candidates: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("gtkw"))
                    .unwrap_or(false)
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

// From a non-empty list, return the single entry directly, or prompt the user to
// pick one (labels shown relative to `root` so entries from different runs are
// distinguishable). Ok(None) means the user cancelled the selection.
fn pick_one<'a>(
    files: &'a [PathBuf],
    root: &Path,
    prompt: &str,
) -> Result<Option<&'a PathBuf>, io::Error> {
    if files.len() == 1 {
        return Ok(Some(&files[0]));
    }

    let labels: Vec<String> = files
        .iter()
        .map(|p| {
            p.strip_prefix(root)
                .unwrap_or(p)
                .to_str()
                .unwrap_or("<unknown>")
                .to_string()
        })
        .collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .items(&labels)
        .default(0)
        .interact_opt()
        .map_err(|e| io::Error::other(format!("Selection failed: {}", e)))?;

    Ok(selection.map(|i| &files[i]))
}

// Print assembly instructions to stdout, syntax-highlighted when writing to a
// terminal. Coloring is disabled when stdout is not a TTY (piped/redirected) or
// when NO_COLOR is set, so downstream tools still receive clean, plain text.
fn print_asm(content: &str) {
    let use_color = io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none();
    if !use_color {
        print!("{}", content);
        if !content.ends_with('\n') {
            println!();
        }
        return;
    }
    for line in content.lines() {
        println!("{}", colorize_asm_line(line));
    }
}

// Token classes recognized in an instr.asm line. The format is line-oriented:
// "cell (row=0, col=0)" headers and "mnemonic(param=\"str\", param=42)"
// instruction lines.
#[derive(PartialEq)]
enum Token {
    Ident,
    Number,
    StringLit,
    Punct,
    Space,
    Other,
}

// ANSI SGR sequences for the highlighter.
const RESET: &str = "\x1b[0m";
const CELL: &str = "\x1b[1;34m"; // bold blue   — "cell" section keyword
const MNEMONIC: &str = "\x1b[1;32m"; // bold green  — instruction opcode
const PARAM: &str = "\x1b[36m"; // cyan        — parameter names
const NUMBER: &str = "\x1b[35m"; // magenta     — integer immediates
const STRING: &str = "\x1b[33m"; // yellow      — quoted values (e.g. variant)
const PUNCT: &str = "\x1b[2m"; // dim         — punctuation ( ) , =

// Syntax-highlight one line of instr.asm. Identifiers are colored by role: the
// "cell" keyword, an opcode (identifier followed by '('), or a parameter name
// (identifier followed by '='); anything else is left uncolored.
fn colorize_asm_line(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();

    // Lex the line into (text, class) tokens.
    let mut tokens: Vec<(String, Token)> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let start = i;
        let class = if c.is_whitespace() {
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            Token::Space
        } else if c == '"' {
            i += 1;
            while i < n && chars[i] != '"' {
                i += 1;
            }
            if i < n {
                i += 1; // include the closing quote
            }
            Token::StringLit
        } else if c.is_ascii_alphabetic() || c == '_' {
            while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            Token::Ident
        } else if c.is_ascii_digit() || (c == '-' && i + 1 < n && chars[i + 1].is_ascii_digit()) {
            i += 1;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            Token::Number
        } else if matches!(c, '(' | ')' | ',' | '=') {
            i += 1;
            Token::Punct
        } else {
            i += 1;
            Token::Other
        };
        tokens.push((chars[start..i].iter().collect(), class));
    }

    // Emit each token wrapped in the color for its (context-dependent) role.
    let mut out = String::new();
    for (idx, (text, class)) in tokens.iter().enumerate() {
        match class {
            Token::StringLit => wrap(&mut out, STRING, text),
            Token::Number => wrap(&mut out, NUMBER, text),
            Token::Punct => wrap(&mut out, PUNCT, text),
            Token::Ident => {
                let next = tokens[idx + 1..]
                    .iter()
                    .find(|(_, k)| *k != Token::Space)
                    .map(|(t, _)| t.as_str());
                if text == "cell" {
                    wrap(&mut out, CELL, text);
                } else if next == Some("(") {
                    wrap(&mut out, MNEMONIC, text);
                } else if next == Some("=") {
                    wrap(&mut out, PARAM, text);
                } else {
                    out.push_str(text);
                }
            }
            Token::Space | Token::Other => out.push_str(text),
        }
    }
    out
}

fn wrap(out: &mut String, color: &str, text: &str) {
    out.push_str(color);
    out.push_str(text);
    out.push_str(RESET);
}

// Recursively walk `root`, collecting every ".svg"/".png" that lives in a
// directory whose path ends with `suffix` (the configured artifact directory,
// e.g. "debug/timetable" or "debug/interconnect"). file_type() is used instead
// of is_dir() so symlinks are not followed, which avoids cycles.
fn collect_images(root: &Path, suffix: &Path, out: &mut Vec<PathBuf>) -> Result<(), io::Error> {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        // A directory we cannot read (permissions, races) should not abort the
        // whole search; just skip it.
        Err(_) => {
            return Ok(());
        }
    };

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();

        if path.ends_with(suffix) {
            for e in fs::read_dir(&path)? {
                let file = e?.path();
                let is_image = file
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("svg") || ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false);
                if file.is_file() && is_image {
                    out.push(file);
                }
            }
        } else {
            collect_images(&path, suffix, out)?;
        }
    }
    Ok(())
}

// Recursively walk `root`, collecting every file named exactly `filename`.
// file_type() is used instead of is_dir()/is_file() so symlinks are not
// followed, which avoids cycles.
fn collect_named(root: &Path, filename: &str, out: &mut Vec<PathBuf>) -> Result<(), io::Error> {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        // A directory we cannot read (permissions, races) should not abort the
        // whole search; just skip it.
        Err(_) => {
            return Ok(());
        }
    };

    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_named(&path, filename, out)?;
        } else if file_type.is_file()
            && path.file_name().and_then(|n| n.to_str()) == Some(filename)
        {
            out.push(path);
        }
    }
    Ok(())
}

// Collapse each timetable's ".svg"/".png" pair to a single entry, preferring the
// SVG. Grouping is by the extension-less path, so a "schedule_rb1.svg" hides its
// sibling "schedule_rb1.png" while unpaired files are kept as-is. The result is
// sorted for a stable, predictable ordering.
fn prefer_svg(images: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut chosen: std::collections::HashMap<PathBuf, PathBuf> = std::collections::HashMap::new();
    for image in images {
        let key = image.with_extension("");
        let is_svg = image
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("svg"))
            .unwrap_or(false);
        match chosen.get(&key) {
            // An SVG always wins over a previously seen PNG; otherwise keep the
            // first candidate for this timetable.
            Some(existing) if !is_svg || existing == &image => {}
            _ => {
                chosen.insert(key, image);
            }
        }
    }

    let mut result: Vec<PathBuf> = chosen.into_values().collect();
    result.sort();
    result
}

// Open a file in an image viewer using the default Linux opener (xdg-open).
fn open_in_viewer(path: &Path) -> Result<(), io::Error> {
    open_with("xdg-open", "the xdg-utils package", &[path.as_os_str()])
}

// Spawn `program` with `args`, detached from our stdio. A missing program yields
// a clear, actionable error (mentioning `install_hint`) instead of failing
// silently; any other spawn error is surfaced too.
fn open_with(program: &str, install_hint: &str, args: &[&OsStr]) -> Result<(), io::Error> {
    let shown = args
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    let result = process::Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    match result {
        Ok(_) => {
            info!("Opening {} with {}", shown, program);
            Ok(())
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            error!(
                "`{}` is not available on this system. Please install it (e.g. {}) \
                 or open the file manually: {}",
                program, install_hint, shown
            );
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{} is not available", program),
            ))
        }
        Err(e) => {
            error!("Failed to open {} with {}: {}", shown, program, e);
            Err(e)
        }
    }
}
