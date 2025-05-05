mod arch_visual_gen;
mod drra;
mod isa;
mod isa_gen;
mod rtl_code_gen;
mod sst_sim_gen;
mod utils;

use clap::{error::ErrorKind, Parser, Subcommand};
use log::{error, info};
use std::fs;
use std::io::Result;
use std::path::Path;

#[derive(Subcommand)]
enum Command {
    #[command(about = "Assemble the system", name = "assemble")]
    Assemble {
        /// Architecture JSON file path
        #[arg(short, long)]
        arch: String,
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
#[command(version, about, long_about=None)]
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
            ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
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
            arch,
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
            assemble(arch, output);
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
        panic!("JSON file is not valid");
    }
}

fn assemble(arch: &String, output: &String) {
    fs::create_dir_all(output).expect("Failed to create output directory");
    fs::create_dir_all(Path::new(output).join("arch")).expect("Failed to create arch directory");
    fs::create_dir_all(Path::new(output).join("isa")).expect("Failed to create isa directory");
    fs::create_dir_all(Path::new(output).join("rtl")).expect("Failed to create rtl directory");
    fs::create_dir_all(Path::new(output).join("sst")).expect("Failed to create sst directory");
    match rtl_code_gen::gen_rtl(&arch, &output, &format!("{}/arch/arch.json", output)) {
        Ok(_) => (),
        Err(e) => panic!("Error: {}", e),
    }
    isa_gen::generate(
        &format!("{}/arch/arch.json", output),
        &format!("{}/isa", output),
    );
    arch_visual_gen::generate(
        &format!("{}/arch/arch.json", output),
        &format!("{}/arch", output),
    );
    sst_sim_gen::generate(
        &format!("{}/arch/arch.json", output),
        &format!("{}/sst", output),
    );

    // Remove write permissions from the output directory
    info!("Removing write permissions from output directory...");
    match remove_write_permissions(output) {
        Ok(_) => info!("Output directory is now read-only"),
        Err(e) => error!("Failed to remove write permissions: {}", e),
    }

    fn remove_write_permissions(dir_path: &str) -> Result<()> {
        fn set_readonly_recursive(path: &Path) -> Result<()> {
            let metadata = fs::metadata(path)?;
            let mut perms = metadata.permissions();

            // Make read-only
            // Only make specific file types read-only
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if let Some(ext_str) = extension.to_str() {
                        let ext_lower = ext_str.to_lowercase();
                        // Only set read-only for json, systemverilog, vhdl and yaml files
                        if ["json", "sv", "vhdl", "vhd", "yaml", "yml"]
                            .contains(&ext_lower.as_str())
                        {
                            perms.set_readonly(true);
                            fs::set_permissions(path, perms)?;
                        }
                    }
                }
            }

            if metadata.is_dir() {
                for entry in fs::read_dir(path)? {
                    let entry = entry?;
                    set_readonly_recursive(&entry.path())?;
                }
            }

            Ok(())
        }

        // Only change permissions of contents, not the directory itself
        let path = Path::new(dir_path);
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            set_readonly_recursive(&entry.path())?;
        }

        Ok(())
    }
}
