mod asm;
mod gen;
mod instr;
mod parser;
use clap::Parser as clappar;
use gen::Generator;
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
    let args = Args::parse();

    let parser = Parser::new(&args.arch, &args.isa);
    let instructions = parser.parse(&args.asm);

    let generator = Generator::new();
    let bin_str = generator.gen_bin(&instructions);
    let txt_str = generator.gen_txt(&instructions);

    // create output folder if it does not exist
    let output_path = Path::new(&args.output);
    if let Err(e) = fs::create_dir_all(output_path) {
        eprintln!("Failed to create output directory: {}", e);
        return;
    }

    // Write the binary and text output to files in the output directory
    let bin_file_path = output_path.join("instr.bin");
    let txt_file_path = output_path.join("instr.txt");

    if let Err(e) = fs::write(&bin_file_path, bin_str) {
        eprintln!("Failed to write binary file: {}", e);
        return;
    }

    if let Err(e) = fs::write(&txt_file_path, txt_str) {
        eprintln!("Failed to write text file: {}", e);
        return;
    }
}
