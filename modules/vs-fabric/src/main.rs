mod arch_visual_gen;
mod isa_gen;
mod models;
mod sst_sim_gen;
mod utils;

mod fabric_manager;
mod loader;
mod resolver;
mod rtl_generator;

use crate::fabric_manager::FabricManager;

use log::{error, info};
use std::{io::Result, path::Path};

use clap::{error::ErrorKind, Parser, Subcommand};

#[derive(Subcommand)]
enum Command {
    #[command(
        about = "Elaborate the fabric: the arch.json and ISA the compiler reads",
        name = "elaborate"
    )]
    Elaborate {
        /// Source architecture JSON file path
        #[arg(short, long)]
        arch_json: String,
        /// Output directory path
        #[arg(short, long)]
        output: String,
        // Debug mode (default: false)
        #[arg(short, long, default_value_t = false)]
        debug: bool,
    },
    #[command(
        about = "Generate the SST simulation files from an elaborated or sized arch.json",
        name = "sst"
    )]
    Sst {
        /// Elaborated or sized architecture JSON file path
        #[arg(short, long)]
        arch_json: String,
        /// Output directory path
        #[arg(short, long)]
        output: String,
        // Debug mode (default: false)
        #[arg(short, long, default_value_t = false)]
        debug: bool,
    },
    #[command(
        about = "Generate the RTL from an elaborated or sized arch.json",
        name = "rtl"
    )]
    Rtl {
        /// Elaborated or sized architecture JSON file path
        #[arg(short, long)]
        arch_json: String,
        /// Output directory path
        #[arg(short, long)]
        output: String,
        // Debug mode (default: false)
        #[arg(short, long, default_value_t = false)]
        debug: bool,
    },
}

#[derive(Parser)]
#[command(about, long_about=None)]
struct Args {
    /// Command to execute
    #[command(subcommand)]
    command: Command,
}

fn main() {
    // Set the log level
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();
    log_panics::init();

    // Make sure the program returns non-zero if command parsing fails
    let cli_args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => match e.kind() {
            ErrorKind::DisplayHelp => {
                println!("{}", e);
                std::process::exit(0);
            }
            _ => {
                error!("{}", e);
                std::process::exit(1);
            }
        },
    };

    let (step, arch_json, output, debug) = match &cli_args.command {
        Command::Elaborate {
            arch_json,
            output,
            debug,
        } => ("elaborate", arch_json, output, debug),
        Command::Sst {
            arch_json,
            output,
            debug,
        } => ("sst", arch_json, output, debug),
        Command::Rtl {
            arch_json,
            output,
            debug,
        } => ("rtl", arch_json, output, debug),
    };

    let debug_level = if *debug {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    env_logger::builder().filter_level(debug_level);

    info!("Running {} on {} into {}", step, arch_json, output);

    if let Err(e) = run(&cli_args.command, arch_json, output) {
        error!("{} failed: {}", step, e);
        std::process::exit(1);
    }
    info!("{} completed successfully!", step);
}

fn run(command: &Command, arch_json: &str, output: &str) -> Result<()> {
    let manager = FabricManager::new(output)?;
    let arch_json = Path::new(arch_json);
    match command {
        Command::Elaborate { .. } => manager.elaborate(arch_json),
        Command::Sst { .. } => manager.generate_sst(arch_json),
        Command::Rtl { .. } => manager.generate_rtl(arch_json),
    }
}
