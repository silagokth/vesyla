mod arch_visual_gen;
mod isa;
mod isa_gen;
mod models;
mod rtl_code_gen;
mod sst_sim_gen;
mod utils;

use crate::utils::remove_write_permissions;

use log::{error, info};
use std::{fs, io::Result, path::Path};

use clap::{error::ErrorKind, Parser, Subcommand};

#[derive(Subcommand)]
enum Command {
    #[command(about = "Assemble the system", name = "assemble")]
    Assemble {
        /// Architecture JSON file path
        #[arg(short, long)]
        arch_json: String,
        /// Output directory path
        #[arg(short, long)]
        output: String,
        // Debug mode (default: false)
        #[arg(short, long, default_value_t = false)]
        debug: bool,
    },
    #[command(about = "Validate JSON file", name = "validate_json")]
    ValidateJson {
        /// Path to the JSON file
        #[arg(short, long)]
        json_file: String,
        #[arg(short, long)]
        schema_file: String,
    },
    #[command(about = "Clean the build directory", name = "clean")]
    Clean {
        /// Build directory
        #[arg(short, long, default_value = "build")]
        build_dir: String,
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
    // set the log level
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();
    log_panics::init();

    // make sure the program return non-zero if command parsing fails
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

    match &cli_args.command {
        Command::Assemble {
            arch_json,
            output,
            debug,
        } => {
            let _debug_level = if *debug {
                log::LevelFilter::Debug
            } else {
                log::LevelFilter::Info
            };
            env_logger::builder().filter_level(_debug_level);
            info!("Assembling ...");
            assemble(arch_json, output);
            info!("Done!");
        }
        Command::ValidateJson {
            json_file,
            schema_file,
        } => {
            info!("Validating JSON file ...");
            match validate_json(json_file.clone(), schema_file.clone()) {
                Ok(_) => info!("Done!"),
                Err(e) => error!("Error: {}", e),
            };
        }
        Command::Clean { build_dir } => {
            info!("Cleaning build directory ...");
            let rtl_output_dir = rtl_code_gen::get_rtl_output_dir(build_dir).unwrap();
            match clean(rtl_output_dir.to_str().unwrap().to_string()) {
                Ok(_) => info!("Done!"),
                Err(e) => error!("Error:  --output <OUTPUT>{}", e),
            };
        }
    }
}

fn clean(build_dir: String) -> Result<()> {
    let build_dir = Path::new(&build_dir);
    if build_dir.exists() {
        fs::remove_dir_all(build_dir)?;
    }
    Ok(())
}

fn validate_json(json_file: String, schema_file: String) -> Result<()> {
    let json_file = match fs::File::open(Path::new(&json_file)) {
        Ok(file) => serde_json::from_reader(file).expect("Failed to parse JSON file"),
        Err(err) => {
            println!("Error: {}", err);
            panic!("Failed to open file: {}", json_file);
        }
    };
    let schema_file = match fs::File::open(Path::new(&schema_file)) {
        Ok(file) => serde_json::from_reader(file).expect("Failed to parse schema file"),
        Err(err) => {
            println!("Error: {}", err);
            panic!("Failed to open file: {}", schema_file);
        }
    };
    let validator = jsonschema::validator_for(&schema_file).expect("Failed to create validator");
    if validator.is_valid(&json_file) {
        info!("JSON file is valid");
        Ok(())
    } else {
        for error in validator.iter_errors(&json_file) {
            error!("Validation error: {}", error);
        }
        panic!("JSON file is not valid");
    }
}

fn assemble(arch_json_path_str: &String, output: &String) {
    let arch_output_path = Path::new(output).join("arch");
    let sst_output_path = Path::new(output).join("sst");
    let doc_output_path = Path::new(output).join("isa/");
    let rtl_output_path = Path::new(output).join("rtl/");
    fs::create_dir_all(output).expect("Failed to create output directory");
    fs::create_dir_all(&arch_output_path).expect("Failed to create arch directory");
    fs::create_dir_all(&doc_output_path).expect("Failed to create isa directory");
    fs::create_dir_all(&rtl_output_path).expect("Failed to create rtl directory");
    fs::create_dir_all(&sst_output_path).expect("Failed to create sst directory");

    let arch_json_path = Path::new(arch_json_path_str);
    let arch_output_file = Path::new(&arch_output_path).join("arch.json");
    match rtl_code_gen::gen_rtl(arch_json_path, output, Some(&arch_output_file)) {
        Ok(_) => (),
        Err(e) => panic!("Error: {}", e),
    }

    // Generate ISA (doc and json) from architecture JSON file
    isa_gen::generate(&arch_output_file, &doc_output_path);
    arch_visual_gen::generate(&arch_output_file, &arch_output_path);
    sst_sim_gen::generate(&arch_output_file, &sst_output_path);

    // Remove write permissions from the output directory
    info!("Removing write permissions from output directory...");
    match remove_write_permissions(output) {
        Ok(_) => info!("Output directory is now read-only"),
        Err(e) => error!("Failed to remove write permissions: {}", e),
    }
}
