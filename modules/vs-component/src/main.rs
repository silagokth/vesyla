mod component_generator;
mod utils;

use crate::component_generator::ComponentGenerator;

use log::{error, info};
use std::{fs, io::Result, path::Path};

use clap::{error::ErrorKind, Parser, Subcommand};

#[derive(Subcommand)]
enum Command {
    #[command(about = "Create a new component", name = "create")]
    Create {
        /// Architecture JSON file path
        #[arg(short, long)]
        arch_json: String,
        /// Instruction Set JSON file path
        #[arg(short, long)]
        isa_json: String,
        // Output directory path for the new component
        #[arg(short, long)]
        output_dir: String,
        // Force overwrite if the output directory exists
        #[arg(short, long, default_value_t = false)]
        force: bool,
        // Non interactive mode
        #[arg(short, long, default_value_t = false)]
        non_interactive: bool,
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

    match &cli_args.command {
        Command::Create {
            arch_json,
            isa_json,
            output_dir,
            force,
            non_interactive,
        } => {
            info!(
                "Creating new component with arch: {}, isa: {}, output: {}",
                arch_json, isa_json, output_dir
            );
            match ComponentGenerator::create(
                Path::new(arch_json),
                Path::new(isa_json),
                Path::new(output_dir),
                *force,
                *non_interactive,
            ) {
                Ok(_) => info!("Component creation completed successfully!"),
                Err(e) => {
                    error!("Component creation failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Command::ValidateJson {
            json_file,
            schema_file,
        } => {
            info!("Validating JSON file: {}", json_file);
            match validate_json(json_file, schema_file) {
                Ok(_) => info!("JSON validation completed successfully!"),
                Err(e) => {
                    error!("JSON validation failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Command::Clean { build_dir } => {
            info!("Cleaning build directory: {}", build_dir);
            match clean(build_dir) {
                Ok(_) => info!("Clean completed successfully!"),
                Err(e) => {
                    error!("Clean failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

fn clean(build_dir: &str) -> Result<()> {
    let build_path = Path::new(build_dir);
    if build_path.exists() {
        info!("Cleaning build directory: {}", build_dir);
        fs::remove_dir_all(build_path)?;
        info!("Build directory cleaned");
    } else {
        info!("Build directory does not exist: {}", build_dir);
    }
    Ok(())
}

fn validate_json(json_file: &str, schema_file: &str) -> Result<()> {
    let json_content = fs::File::open(Path::new(json_file))
        .map_err(|e| {
            error!("Failed to open JSON file '{}': {}", json_file, e);
            e
        })
        .and_then(|file| {
            serde_json::from_reader(file).map_err(|e| {
                error!("Failed to parse JSON file '{}': {}", json_file, e);
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })
        })?;

    let schema_content = fs::File::open(Path::new(schema_file))
        .map_err(|e| {
            error!("Failed to open schema file '{}': {}", schema_file, e);
            e
        })
        .and_then(|file| {
            serde_json::from_reader(file).map_err(|e| {
                error!("Failed to parse schema file '{}': {}", schema_file, e);
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })
        })?;

    let validator = jsonschema::validator_for(&schema_content).map_err(|e| {
        error!("Failed to create validator from schema: {}", e);
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    })?;

    if validator.is_valid(&json_content) {
        info!("JSON file is valid according to schema");
        Ok(())
    } else {
        error!("JSON validation errors:");
        for validation_error in validator.iter_errors(&json_content) {
            error!("  - {}", validation_error);
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "JSON file does not conform to schema",
        ))
    }
}
