use crate::instr::Instruction;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

pub struct Generator {}

impl Generator {
    pub fn new() -> Generator {
        Generator {}
    }

    pub fn gen_bin(&self, instructions: &HashMap<String, Vec<Instruction>>) -> String {
        let mut bin_str = String::new();
        for cell_instr in instructions {
            let cell_label = cell_instr.0;
            let instr_list = cell_instr.1;
            let cell_label_split = cell_label.split("_").collect::<Vec<&str>>();
            let x = cell_label_split[0].parse::<i64>().unwrap();
            let y = cell_label_split[1].parse::<i64>().unwrap();
            bin_str = bin_str + &format!("cell {} {}\n", x, y);
            for instruction in instr_list.iter() {
                let bin = instruction.to_bin();
                bin_str = bin_str + &format!("{}\n", bin);
            }
        }
        bin_str
    }

    pub fn gen_txt(&self, instructions: &HashMap<String, Vec<Instruction>>) -> String {
        let mut txt_str = String::new();
        for cell_instr in instructions {
            let cell_label = cell_instr.0;
            let instr_list = cell_instr.1;
            let cell_label_split = cell_label.split("_").collect::<Vec<&str>>();
            let x = cell_label_split[0].parse::<i64>().unwrap();
            let y = cell_label_split[1].parse::<i64>().unwrap();
            txt_str = txt_str + &format!("cell (x={}, y={})\n", x, y);
            for instruction in instr_list.iter() {
                let txt = instruction.to_txt();
                txt_str = txt_str + &format!("{}\n", txt);
            }
        }
        txt_str
    }
}
