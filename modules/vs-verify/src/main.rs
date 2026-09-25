//! Verify one testcase: assemble the fabric, run the C++ reference, compile and
//! simulate the program, and check that the simulators agree with the
//! reference.
//!
//! These are the stages the testcase template's `run.sh` walks, driven from
//! here and shelling out to the same helper scripts under `scripts/`, so the
//! two cannot drift in what they actually invoke. `run.sh` is untouched and
//! `vesyla testcase run` -- and with it the Robot suite -- still goes through
//! it. What this adds is a preflight pass over the external programs each stage
//! needs, so a missing simulator is named before anything runs instead of
//! surfacing as a generic stage failure, and `--skip`, so the compiler can be
//! exercised without waiting for a simulator that is not the thing under test.
//!
//! It verifies the testcase in the current directory unless told otherwise, and
//! `--mlir` or `--pasm` points it at one program in particular rather than the
//! one the testcase carries -- between them, what a compiler change needs to be
//! tried out, without going through the suite.

use clap::{Parser, ValueEnum};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// Exit codes, as run.sh set them. The generated Robot suite decodes these into
// named steps -- Setup, Model 2 run, Model 2 output, RTL run, RTL output -- so
// they are an interface rather than an implementation detail, and a testcase
// run through either entry point has to fail the same way.
const EXIT_SETUP: i32 = 1;
const EXIT_SST_RUN: i32 = 2;
const EXIT_SST_MISMATCH: i32 = 3;
const EXIT_RTL_RUN: i32 = 4;
const EXIT_RTL_MISMATCH: i32 = 5;

// The word width the memory images are decoded at when reporting a mismatch.
// dump_sram_image.py defaults to the same, and a row is read most significant
// word first, the way that script chunks it.
const WORD_BITS: usize = 16;

// How many differing addresses a mismatch report prints before summarising the
// rest. A whole image differing is the common case when something is badly
// wrong, and pages of it bury the first address that went astray.
const MAX_REPORTED_DIFFERENCES: usize = 16;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Skippable {
    /// Instruction-level simulation (model 2) and its check against model 0.
    Sst,
    /// RTL simulation (model 3) and its check against model 0.
    Rtl,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Interactive {
    All,
    Sst,
    Rtl,
}

#[derive(Parser)]
#[command(
    about = "Run a testcase through every model and check that they agree",
    long_about = None
)]
struct Args {
    /// Testcase directory
    #[arg(short, long, default_value = ".")]
    directory: String,

    /// Directory to lay the testcase out in and run it from
    #[arg(short, long, default_value = "out")]
    output_dir: String,

    /// Compile this MLIR program instead of the testcase's own
    #[arg(long, conflicts_with = "pasm")]
    mlir: Option<String>,

    /// Compile this PASM program instead of the testcase's own
    #[arg(long)]
    pasm: Option<String>,

    /// Template directory (defaults to the installed testcase template)
    #[arg(short, long)]
    template_dir: Option<String>,

    /// Stage to skip; pass twice to skip both
    #[arg(long, value_enum)]
    skip: Vec<Skippable>,

    /// Open a simulator interactively rather than running it batch
    #[arg(long, value_enum)]
    interactive: Option<Interactive>,

    /// Show every subprocess as it runs, and capture simulator traces
    #[arg(long)]
    debug: bool,

    /// Never colour the output
    #[arg(long)]
    no_color: bool,
}

//===----------------------------------------------------------------------===//
// Console output
//===----------------------------------------------------------------------===//

struct Ui {
    color: bool,
}

impl Ui {
    // Colour is on for a terminal that has not asked otherwise. NO_COLOR is
    // honoured because this runs under the Robot suite and in CI logs, where
    // escape codes are noise.
    fn new(requested: bool) -> Ui {
        let color = requested
            && std::io::stdout().is_terminal()
            && env::var_os("NO_COLOR").is_none();
        Ui { color }
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if !self.color {
            return text.to_string();
        }
        format!("\x1b[{}m{}\x1b[0m", code, text)
    }

    fn stage(&self, name: &str) {
        println!("{}", self.paint("1", name));
    }

    fn step(&self, what: &str) {
        print!("  {} ... ", what);
        let _ = std::io::stdout().flush();
    }

    fn ok(&self) {
        println!("{}", self.paint("0;32", "ok"));
    }

    fn skipped(&self, what: &str, why: &str) {
        println!("  {} ... {} ({})", what, self.paint("0;33", "skipped"), why);
    }

    fn note(&self, what: &str) {
        println!("  {} {}", self.paint("0;33", "warning:"), what);
    }

    // Report a failed step and stop. The code is the one run.sh would have
    // exited with, so the Robot suite reads it the same way either way.
    fn fatal(&self, code: i32, detail: &str) -> ! {
        println!("{}", self.paint("0;31", "FAILED"));
        eprintln!("{}", detail.trim_end());
        std::process::exit(code);
    }

    // A failure with no step in progress -- preflight and layout, which run
    // before the first stage header is printed.
    fn abort(&self, code: i32, detail: &str) -> ! {
        eprintln!("{} {}", self.paint("0;31", "error:"), detail.trim_end());
        std::process::exit(code);
    }
}

//===----------------------------------------------------------------------===//
// Preflight
//===----------------------------------------------------------------------===//

// The first directory on PATH holding an executable of this name.
fn which(program: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for directory in env::split_paths(&path) {
        let candidate = directory.join(program);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// The programs the stages that are actually going to run will shell out to.
//
// Every one of these is reached from inside a helper script, several of them
// minutes into a run, so without this a missing simulator or an unset component
// path shows up as a subprocess failure with the reason buried in a captured
// log. Only what the selected stages need is required: skipping the RTL stage
// should not ask for Questa.
fn preflight(run_sst: bool, run_rtl: bool) -> Result<(), String> {
    let mut required: Vec<(&str, &str)> = vec![
        (
            "vesyla",
            "the helper scripts call it to assemble the fabric and compile the program",
        ),
        ("g++", "builds the model 0 C++ reference"),
        (
            "python3",
            "dumps the memory images to .hex/.txt next to each .bin",
        ),
    ];
    if run_sst {
        required.push(("sst", "runs the instruction-level model (model 2)"));
    }
    if run_rtl {
        required.push(("bender", "gathers the RTL sources for model 3"));
        required.push(("vsim", "runs the RTL simulation (model 3)"));
    }

    let mut missing = Vec::new();
    for (program, why) in required {
        if which(program).is_none() {
            missing.push(format!("  {:<8} not on PATH -- {}", program, why));
        }
    }

    // The component library is what the fabric is assembled out of and what
    // instruction selection reads its patterns from, so it is as much a
    // requirement as any of the programs above.
    match env::var("VESYLA_SUITE_PATH_COMPONENTS") {
        Err(_) => {
            missing.push(
                "  VESYLA_SUITE_PATH_COMPONENTS is unset -- it has to point at a built \
                 component library"
                    .to_string(),
            );
        }
        Ok(path) => {
            if !Path::new(&path).join("resources").is_dir() {
                missing.push(format!(
                    "  VESYLA_SUITE_PATH_COMPONENTS is {}, which has no resources/ -- is it a \
                     built component library?",
                    path
                ));
            }
        }
    }

    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the tools these stages need are not all available:\n{}",
        missing.join("\n")
    ))
}

//===----------------------------------------------------------------------===//
// Laying the testcase out
//===----------------------------------------------------------------------===//

// A sibling executable of this one, the way the vesyla driver finds its tools.
fn sibling(program: &str) -> PathBuf {
    match env::current_exe().ok().and_then(|exe| exe.parent().map(|dir| dir.join(program))) {
        Some(path) => path,
        None => PathBuf::from(program),
    }
}

// Copy a tree, leaving `exclude` behind wherever it turns up.
//
// That exclusion is what lets the output directory sit inside the testcase,
// which is where it lands by default when this is run from a testcase
// directory: without it, copying the testcase would copy the output directory
// into itself, and go on doing so.
fn copy_dir_all(source: &Path, destination: &Path, exclude: &Path) -> Result<(), String> {
    fs::create_dir_all(destination)
        .map_err(|e| format!("cannot create {}: {}", destination.display(), e))?;
    let entries = fs::read_dir(source)
        .map_err(|e| format!("cannot read {}: {}", source.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read {}: {}", source.display(), e))?;
        if entry.path() == exclude {
            continue;
        }
        let target = destination.join(entry.file_name());
        let kind = entry
            .file_type()
            .map_err(|e| format!("cannot stat {}: {}", entry.path().display(), e))?;
        if kind.is_dir() {
            copy_dir_all(&entry.path(), &target, exclude)?;
        } else {
            fs::copy(entry.path(), &target)
                .map_err(|e| format!("cannot copy {}: {}", entry.path().display(), e))?;
        }
    }
    Ok(())
}

// The program to verify and the extension it has to be written under, when one
// was named on the command line rather than taken from the testcase.
fn requested_program(args: &Args) -> Result<Option<(PathBuf, &'static str)>, String> {
    let (given, extension) = match (&args.mlir, &args.pasm) {
        (Some(path), _) => (path, "mlir"),
        (_, Some(path)) => (path, "pasm"),
        _ => return Ok(None),
    };
    let program = fs::canonicalize(given)
        .map_err(|e| format!("cannot open program {}: {}", given, e))?;
    if !program.is_file() {
        return Err(format!("{} is not a file", program.display()));
    }
    Ok(Some((program, extension)))
}

// Lay the template down and the testcase over it, which is what makes a
// testcase runnable: the testcase carries only what is specific to it
// (arch.json, model_0, pasm), and everything else -- run.sh, the helper scripts
// -- comes from the template. `vs-testcase init` owns where the installed
// template lives, so it is asked rather than reimplemented.
fn lay_out(args: &Args) -> Result<PathBuf, String> {
    let testcase = fs::canonicalize(&args.directory)
        .map_err(|e| format!("cannot open testcase {}: {}", args.directory, e))?;
    if !testcase.is_dir() {
        return Err(format!("{} is not a directory", testcase.display()));
    }

    fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("cannot create {}: {}", args.output_dir, e))?;
    let root = fs::canonicalize(&args.output_dir)
        .map_err(|e| format!("cannot open {}: {}", args.output_dir, e))?;
    if root == testcase {
        return Err(format!(
            "the output directory is the testcase itself ({}) -- laying the template down there \
             would write run.sh, scripts/ and anything else the testcase does not carry into it; \
             pass -o with somewhere else",
            root.display()
        ));
    }

    // Read before the template is laid down, so a path that is not there is
    // reported before anything has been written.
    let program = requested_program(args)?;

    let mut init = Command::new(sibling("vs-testcase"));
    init.args(["init", "--force", "--output"]);
    init.arg(&root);
    if let Some(template) = &args.template_dir {
        init.args(["--template-dir", template]);
    }
    run(init, false)
        .map_err(|detail| format!("laying down the testcase template failed\n{}", detail))?;

    copy_dir_all(&testcase, &root, &root)?;

    // A program named on the command line replaces the testcase's own, written
    // in as program 0: that is the only one anything downstream simulates --
    // instr_sim.sh and rtl_sim.sh are both called with id 0 -- and compile.sh
    // chooses the MLIR or the PASM path by which of pasm/0.mlir and pasm/0.pasm
    // it finds, so the extension is what selects the front end.
    if let Some((source, extension)) = program {
        let programs = root.join("pasm");
        if programs.exists() {
            fs::remove_dir_all(&programs)
                .map_err(|e| format!("cannot clear {}: {}", programs.display(), e))?;
        }
        fs::create_dir_all(&programs)
            .map_err(|e| format!("cannot create {}: {}", programs.display(), e))?;
        let target = programs.join(format!("0.{}", extension));
        fs::copy(&source, &target).map_err(|e| {
            format!("cannot copy {} to {}: {}", source.display(), target.display(), e)
        })?;
    }

    Ok(root)
}

//===----------------------------------------------------------------------===//
// Running things
//===----------------------------------------------------------------------===//

// Run a command, returning its combined output as the error when it fails.
//
// In debug mode the subprocess keeps this process's streams instead, so a
// simulator's progress is visible as it happens rather than arriving in one
// block at the end.
fn run(mut command: Command, verbose: bool) -> Result<(), String> {
    let program = command.get_program().to_string_lossy().into_owned();

    if verbose {
        let status = command
            .status()
            .map_err(|e| format!("cannot run {}: {}", program, e))?;
        if status.success() {
            return Ok(());
        }
        return Err(format!("exited with status {}", status.code().unwrap_or(-1)));
    }

    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("cannot run {}: {}", program, e))?;
    if output.status.success() {
        return Ok(());
    }
    let mut detail = String::from_utf8_lossy(&output.stdout).into_owned();
    detail.push_str(&String::from_utf8_lossy(&output.stderr));
    Err(detail)
}

// Run one of the testcase's helper scripts from the work directory. They derive
// their own paths from where they sit, but several of them write to paths
// relative to the working directory, so that has to be `work` -- run.sh cds
// there before it calls any of them.
fn script(root: &Path, name: &str, args: &[&str], debug: bool) -> Result<(), String> {
    let mut command = Command::new("bash");
    command.arg(root.join("scripts").join(name));
    command.args(args);
    command.current_dir(root.join("work"));
    if debug {
        // The instruction-level model reads this to turn on per-cycle prints
        // and its JSON trace; leaving it unset keeps the fast path.
        command.env("VESYLA_DEBUG", "1");
    }
    run(command, debug)
}

// Write the .hex and .txt siblings of a memory image. Best effort: the dump is
// for reading afterwards, and losing it is not worth failing a run over, which
// is how run.sh treats it too.
fn dump_image(root: &Path, image: &Path) {
    let mut command = Command::new("python3");
    command
        .arg(root.join("scripts").join("dump_sram_image.py"))
        .arg(image)
        .args(["--data-type", "int16_t"])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let _ = command.status();
}

//===----------------------------------------------------------------------===//
// Memory images
//===----------------------------------------------------------------------===//

// One line of an image: its address and bit pattern, when it has the shape
// dump_sram_image.py recognises -- "<address> <bits>", with anything after the
// bits ignored. A line it does not recognise is not a row; that script skips
// such lines and so does this, so the two read a file the same way.
fn parse_row(line: &str) -> Option<(u64, &str)> {
    let mut fields = line.split_whitespace();
    let address = fields.next()?.parse::<u64>().ok()?;
    let bits = fields.next()?;
    if bits.is_empty() || !bits.chars().all(|c| c == '0' || c == '1') {
        return None;
    }
    Some((address, bits))
}

// Read an image as address -> bits. This is what the SST model and the RTL
// testbench both write, and what model 0 writes as the reference.
fn read_image(path: &Path) -> Result<BTreeMap<u64, String>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

    let mut rows = BTreeMap::new();
    for line in text.lines() {
        if let Some((address, bits)) = parse_row(line.trim()) {
            rows.insert(address, bits.to_string());
        }
    }
    if rows.is_empty() {
        return Err(format!(
            "{} holds no \"<address> <bits>\" rows",
            path.display()
        ));
    }
    Ok(rows)
}

// Rewrite an image trimmed and in address order.
//
// The comparison below does not need this -- it reads rows into a map keyed by
// address -- but run.sh leaves the files sorted and unindented, and they are
// read afterwards both by hand and by dump_sram_image.py, so they are left in
// the same shape here. Every line is kept, including any this cannot order.
fn normalise_image(path: &Path) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

    let mut lines: Vec<(Option<u64>, &str)> = text
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(|line| (parse_row(line).map(|(address, _)| address), line))
        .collect();
    // Stable, so lines with no address keep their order among themselves.
    lines.sort_by_key(|(address, _)| *address);

    let mut sorted = String::new();
    for (_, line) in &lines {
        sorted.push_str(line);
        sorted.push('\n');
    }
    fs::write(path, sorted).map_err(|e| format!("cannot write {}: {}", path.display(), e))
}

// One row's words, decoded the way dump_sram_image.py decodes them: signed,
// most significant word first. A row whose width is not a whole number of words
// is shown as its bits, since splitting it would only invent a reading.
fn decode_row(bits: &str) -> String {
    if bits.len() % WORD_BITS != 0 {
        return bits.to_string();
    }
    let characters: Vec<char> = bits.chars().collect();
    let mut words = Vec::new();
    for chunk in characters.chunks(WORD_BITS) {
        let text: String = chunk.iter().collect();
        let magnitude = match i64::from_str_radix(&text, 2) {
            Ok(value) => value,
            Err(_) => return bits.to_string(),
        };
        // Two's complement: the top bit of the word carries the sign.
        let value = if chunk[0] == '1' {
            magnitude - (1i64 << WORD_BITS)
        } else {
            magnitude
        };
        words.push(value.to_string());
    }
    format!("[{}]", words.join(", "))
}

// Compare a simulator's image against the model 0 reference.
//
// run.sh compares with `diff -q`, which says only that the two are not the
// same. Since the format is known, this says which addresses differ and what
// they hold, decoded as int16 -- which is where the investigation would have
// started anyway.
fn compare_images(reference: &Path, actual: &Path) -> Result<(), String> {
    let expected = read_image(reference)?;
    let produced = read_image(actual)?;

    let mut addresses: Vec<u64> = expected.keys().chain(produced.keys()).copied().collect();
    addresses.sort_unstable();
    addresses.dedup();

    let absent = "<absent>".to_string();
    let mut differences = Vec::new();
    for address in &addresses {
        let want = expected.get(address);
        let got = produced.get(address);
        if want == got {
            continue;
        }
        differences.push(format!(
            "  address {}\n    expected {}\n    actual   {}",
            address,
            want.map(|bits| decode_row(bits)).unwrap_or_else(|| absent.clone()),
            got.map(|bits| decode_row(bits)).unwrap_or_else(|| absent.clone())
        ));
    }

    if differences.is_empty() {
        return Ok(());
    }

    let mut detail = format!(
        "{} of {} addresses differ between {} and {}:\n",
        differences.len(),
        addresses.len(),
        reference.display(),
        actual.display()
    );
    for difference in differences.iter().take(MAX_REPORTED_DIFFERENCES) {
        detail.push_str(difference);
        detail.push('\n');
    }
    if differences.len() > MAX_REPORTED_DIFFERENCES {
        detail.push_str(&format!(
            "  ... and {} more\n",
            differences.len() - MAX_REPORTED_DIFFERENCES
        ));
    }
    Err(detail)
}

//===----------------------------------------------------------------------===//
// Stages
//===----------------------------------------------------------------------===//

fn main() {
    let args = Args::parse();
    let ui = Ui::new(!args.no_color);

    let run_sst = !args.skip.contains(&Skippable::Sst);
    let run_rtl = !args.skip.contains(&Skippable::Rtl);

    if let Err(why) = preflight(run_sst, run_rtl) {
        ui.abort(EXIT_SETUP, &why);
    }

    let root = match lay_out(&args) {
        Ok(root) => root,
        Err(why) => ui.abort(EXIT_SETUP, &why),
    };
    let work = root.join("work");
    let mem = work.join("mem");
    let reference = mem.join("sram_image_m0.bin");

    // Setup: a fresh work directory and the fabric assembled into it.
    ui.stage("Setup");
    ui.step("Assembling the fabric");
    if work.exists() {
        if let Err(e) = fs::remove_dir_all(&work) {
            ui.fatal(EXIT_SETUP, &format!("cannot clear {}: {}", work.display(), e));
        }
    }
    if let Err(e) = fs::create_dir_all(&mem) {
        ui.fatal(EXIT_SETUP, &format!("cannot create {}: {}", mem.display(), e));
    }
    if let Err(detail) = script(&root, "assemble.sh", &[], args.debug) {
        ui.fatal(EXIT_SETUP, &detail);
    }
    ui.ok();

    // Model 0: the C++ reference. It writes both the input image every later
    // model reads and the output image they are all checked against.
    ui.stage("Model 0: C++ reference");
    ui.step("Compiling");
    let mut compile = Command::new("g++");
    compile
        .arg("-g")
        .arg(format!("-I{}", root.join("model_0").join("include").display()))
        .args(["-o", "run_model_0"])
        .arg(root.join("model_0").join("main.cpp"))
        .arg(root.join("model_0").join("src").join("Drra.cpp"))
        .arg(root.join("model_0").join("src").join("Util.cpp"))
        .current_dir(&work);
    if let Err(detail) = run(compile, args.debug) {
        ui.fatal(EXIT_SETUP, &detail);
    }
    ui.ok();

    ui.step("Running");
    let mut model_0 = Command::new(work.join("run_model_0"));
    model_0.current_dir(&work);
    if let Err(detail) = run(model_0, args.debug) {
        ui.fatal(EXIT_SETUP, &detail);
    }
    ui.ok();

    ui.step("Checking the reference output");
    match fs::metadata(&reference) {
        Err(_) => {
            ui.fatal(
                EXIT_SETUP,
                &format!("{} was not written by the reference", reference.display()),
            );
        }
        Ok(metadata) => {
            if metadata.len() == 0 {
                ui.fatal(EXIT_SETUP, &format!("{} is empty", reference.display()));
            }
        }
    }
    if let Err(why) = normalise_image(&reference) {
        ui.fatal(EXIT_SETUP, &why);
    }
    dump_image(&root, &reference);
    ui.ok();

    // Model 1 has no implementation; run.sh stands the reference in for it so
    // the artifact exists, and the same is done here.
    ui.stage("Model 1");
    ui.skipped("Running", "not implemented; model 0 stands in");
    let model_1 = mem.join("sram_image_m1.bin");
    if let Err(e) = fs::copy(&reference, &model_1) {
        ui.abort(
            EXIT_SETUP,
            &format!("cannot write {}: {}", model_1.display(), e),
        );
    }
    dump_image(&root, &model_1);

    // Model 2: compile the program, then simulate it. The compile is not
    // skippable -- it is what the RTL stage runs too, and usually the thing
    // being tested.
    ui.stage("Model 2: instruction-level simulation");
    ui.step("Compiling");
    let pasm = root.join("pasm");
    let pasm = match pasm.to_str() {
        Some(path) => path.to_string(),
        None => ui.fatal(EXIT_SST_RUN, "the testcase path is not valid UTF-8"),
    };
    let mut compile_args = vec![pasm.as_str()];
    if args.debug {
        compile_args.push("-d");
    }
    if let Err(detail) = script(&root, "compile.sh", &compile_args, args.debug) {
        ui.fatal(EXIT_SST_RUN, &detail);
    }
    ui.ok();

    let model_2 = mem.join("sram_image_m2.bin");
    if run_sst {
        if matches!(args.interactive, Some(Interactive::All) | Some(Interactive::Sst)) {
            ui.note("interactive mode is not implemented for SST; running batch");
        }
        ui.step("Running");
        if let Err(detail) = script(&root, "instr_sim.sh", &["0"], args.debug) {
            ui.fatal(EXIT_SST_RUN, &detail);
        }
        ui.ok();

        ui.step("Checking against model 0");
        if let Err(why) = normalise_image(&model_2) {
            ui.fatal(EXIT_SST_MISMATCH, &why);
        }
        dump_image(&root, &model_2);
        if let Err(why) = compare_images(&reference, &model_2) {
            ui.fatal(EXIT_SST_MISMATCH, &why);
        }
        ui.ok();
    } else {
        ui.skipped("Running", "--skip sst");
    }

    // Model 3: the RTL simulation.
    ui.stage("Model 3: RTL simulation");
    if run_rtl {
        ui.step("Compiling and running");
        let interactive = match args.interactive {
            Some(Interactive::All) => "-it=all",
            Some(Interactive::Rtl) => "-it=rtl",
            _ => "",
        };
        let mut rtl_args = vec!["0"];
        if args.debug {
            rtl_args.push("-d");
        }
        if !interactive.is_empty() {
            rtl_args.push(interactive);
        }
        if let Err(detail) = script(&root, "rtl_sim.sh", &rtl_args, args.debug) {
            ui.fatal(EXIT_RTL_RUN, &detail);
        }
        ui.ok();

        ui.step("Checking against model 0");
        let model_3 = mem.join("sram_image_m3.bin");
        if let Err(why) = normalise_image(&model_3) {
            ui.fatal(EXIT_RTL_MISMATCH, &why);
        }
        dump_image(&root, &model_3);
        if let Err(why) = compare_images(&reference, &model_3) {
            ui.fatal(EXIT_RTL_MISMATCH, &why);
        }
        ui.ok();
    } else {
        ui.skipped("Compiling and running", "--skip rtl");
    }

    println!();
    println!("{}", ui.paint("0;32", "Every model that ran agrees with the reference."));
}
