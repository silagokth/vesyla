use clap::{error::ErrorKind, Parser, Subcommand};
use log::{error, info, LevelFilter};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::io::Write;
use std::path::PathBuf;

#[derive(Subcommand)]
enum Command {
    #[command(
        about = "Convert DRRA binary instructions to constant data array as C++ header file",
        name = "instr2const"
    )]
    Instr2const {
        /// Input instruction file
        #[arg(short, long)] // Default to empty string
        input: String,
        /// Output C++ header file
        #[arg(short, long)]
        output: String,
    },
    #[command(
        about = "Convert ELF binary file to text based hex file",
        name = "bin2hex"
    )]
    Bin2hex {
        /// Input instruction file
        #[arg(short, long)] // Default to empty string
        input: String,
        /// Memory image size (Number of 32-bit words to be allocated)
        #[arg(short, long, default_value = "32768")]
        size: usize,
        /// Output C++ header file
        #[arg(short, long)]
        output: String,
    },
}

#[derive(Parser)]
#[command(about, long_about = None, allow_missing_positional = true, after_help = "")]
struct Args {
    /// Command to execute
    #[command(subcommand)]
    command: Command,
}

fn main() -> Result<(), io::Error> {
    // set logger level to be debug
    env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .init();

    // make sure the program return non-zero status code when arguments are invalid
    let cli_args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => {
            // Check if the error is for displaying help or version
            match e.kind() {
                ErrorKind::DisplayHelp => {
                    println!("{}", e);
                    return Ok(());
                }
                // For any other parsing error
                _ => {
                    // Log the error message provided by clap
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
        Command::Instr2const { input, output } => {
            if let Err(err) = instr2const(input, output) {
                error!("Failed to execute command instr2const: {:?}", err);
                return Err(err);
            }
            Ok(())
        }
        Command::Bin2hex {
            input,
            size,
            output,
        } => {
            if let Err(err) = bin2hex(input, size, output) {
                error!("Failed to execute command elf2bin: {:?}", err);
                return Err(err);
            }
            Ok(())
        }
    }
}

fn instr2const(input: &String, output: &String) -> Result<(), io::Error> {
    // Check if input is provided
    let input_path = PathBuf::from(input);
    if input.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Input file path is required",
        ));
    }

    // Check if the input file exists
    if !input_path.exists() {
        error!("Input file {:?} does not exist", input_path);
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Input file {:?} does not exist", input_path),
        ));
    }

    // Read the content of the input file
    let content =
        fs::read_to_string(&input_path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // create an map of instructions
    let mut instruction_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut current_cell = String::new();
    for line in content.lines() {
        let trimmed_line = line.trim();
        if !trimmed_line.is_empty() {
            // Split the line into instruction and its arguments
            let parts: Vec<&str> = trimmed_line.split_whitespace().collect();
            if parts.len() > 0 {
                if parts[0] == "cell" && parts.len() == 3 {
                    let row = parts[1].to_string();
                    let col = parts[2].to_string();
                    let label = format!("{}_{}", row, col);
                    current_cell = label;
                    if !instruction_map.contains_key(&current_cell) {
                        instruction_map.insert(current_cell.clone(), Vec::new());
                    }
                } else if parts.len() == 1 {
                    let bin_str = parts[0].to_string();
                    assert_eq!(bin_str.len(), 32, "Binary instruction must be 32 bits long");
                    let hex_str = format!("0x{:08X}", u32::from_str_radix(&bin_str, 2).unwrap());
                    instruction_map
                        .get_mut(&current_cell)
                        .unwrap()
                        .push(hex_str);
                }
            }
        }
    }

    // create a index vector
    let mut index_table: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
    let mut start: usize = 0;
    for (cell, instructions) in &instruction_map {
        let (row, col) = cell.split_once('_').unwrap();
        let row: usize = row.parse().unwrap();
        let col: usize = col.parse().unwrap();

        // the last instruction must be 0x00000000
        if instructions.last() != Some(&"0x00000000".to_string()) {
            error!("The last instruction in cell {} must be 0x00000000", cell);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("The last instruction in cell {} must be 0x00000000", cell),
            ));
        }

        let mut section = 0;
        let mut size = 0;
        for instruction in instructions {
            if instruction != "0x00000000" {
                size += 1;
                continue;
            }
            index_table.push((row, col, section, start, size));
            section += 1;
            start += size;
            size = 0;
        }
    }

    // create instr table
    let mut instr_table: Vec<String> = Vec::new();
    for (_, instructions) in &instruction_map {
        for instruction in instructions {
            instr_table.push(instruction.clone());
        }
    }

    // Write the content to the output file
    let mut output_file =
        File::create(output).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    writeln!(
        output_file,
        "/// Automatic Generated by Vesyla. Do not edit! ///\n"
    )?;
    writeln!(output_file, "#pragma once\n")?;
    writeln!(
        output_file,
        "// Instruction indexing table: (row, column, section, start, size)"
    )?;
    writeln!(
        output_file,
        "static const unsigned int instr_index[][5] = {{"
    )?;
    for (row, col, section, start, size) in index_table {
        writeln!(
            output_file,
            "    {{{}, {}, {}, {}, {}}},",
            row, col, section, start, size
        )?;
    }
    writeln!(output_file, "}};\n")?;
    writeln!(
        output_file,
        "// Instruction data array: 32-bit hex formatted instructions stored continuously."
    )?;
    writeln!(output_file, "static const unsigned int instr_data[] = {{")?;
    for i in 0..instr_table.len() {
        if i != instr_table.len() - 1 {
            writeln!(output_file, "    {},", instr_table[i])?;
        } else {
            // Last element without trailing comma
            writeln!(output_file, "    {}", instr_table[i])?;
        }
    }
    writeln!(output_file, "}};\n")?;

    info!(
        "Successfully converted instructions to constant data array in {}",
        output
    );
    Ok(())
}

fn bin2hex(input: &String, size: &usize, output: &String) -> Result<(), io::Error> {
    // Check if input is provided
    let input_path = PathBuf::from(input);
    if input.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Input file path is required",
        ));
    }

    // Check if the input file exists
    if !input_path.exists() {
        error!("Input file {:?} does not exist", input_path);
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Input file {:?} does not exist", input_path),
        ));
    }

    // read a binary file
    let bin_data = fs::read(&input_path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    assert!(
        bin_data.len() <= *size * 4,
        "Input binary file size exceeds the specified memory image size"
    );
    assert!(
        bin_data.len() % 4 == 0,
        "Input binary file size must be a multiple of 4 bytes"
    );

    // create the output file
    let mut output_file =
        File::create(output).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    for i in 0..*size {
        if i < bin_data.len() / 4 {
            // Read 4 bytes at a time
            let start = i * 4;
            let end = start + 4;
            let word = &bin_data[start..end];
            // Convert to a 32-bit unsigned integer
            let value = u32::from_le_bytes(word.try_into().unwrap());
            // Write to the output file
            writeln!(output_file, "{:08X}", value).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed to write to output file: {}", e),
                )
            })?;
        } else {
            // Fill with zeros if the input is smaller than the size
            writeln!(output_file, "00000000").map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed to write to output file: {}", e),
                )
            })?;
        }
    }

    info!(
        "Successfully converted ELF binary to text based binary in {}",
        output
    );

    Ok(())
}
