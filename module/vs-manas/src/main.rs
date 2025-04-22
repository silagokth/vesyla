mod asm;
mod gen;
mod instr;
mod parser;
use clap::Parser as clappar;
use core::panic;
use gen::Generator;
use log::{debug, error, info, trace, warn};
use parser::Parser;
use std::fs;
use std::path::Path;

#[derive(clappar)]
#[command(version, about, long_about=None)]
struct Args {
    #[arg(short, long)]
    arch: String,
    #[arg(short, long)]
    isa: String,
    #[arg(short = 's', long)]
    asm: String,
    #[arg(short, long, default_value = ".")]
    output: String,
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
    log_panics::init();

    // make sure the program return non-zero if command parsing fails
    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => {
            error!("{}", e);
            panic!();
        }
    };

    info!("Assembler started: {:?}", args.asm);
    let parser = Parser::new(&args.arch, &args.isa);
    let instructions = parser.parse(&args.asm);

    let generator = Generator::new();
    let bin_str = generator.gen_bin(&instructions);
    let txt_str = generator.gen_txt(&instructions);

    // create output folder if it does not exist
    let output_path = Path::new(&args.output);
    if let Err(e) = fs::create_dir_all(output_path) {
        error!("Failed to create output directory: {}", e);
        panic!();
    }

    // Write the binary and text output to files in the output directory
    let bin_file_path = output_path.join("instr.bin");
    let txt_file_path = output_path.join("instr.txt");

    if let Err(e) = fs::write(&bin_file_path, bin_str) {
        error!("Failed to write binary file: {}", e);
        panic!();
    }

    if let Err(e) = fs::write(&txt_file_path, txt_str) {
        error!("Failed to write text file: {}", e);
        panic!();
    }

    info!("Assembler finished!");
    info!("Binary output: {:?}", bin_file_path);
    info!("Text output: {:?}", txt_file_path);
}
