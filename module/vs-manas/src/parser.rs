#![allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use serde_json;
use std::cell;
use std::collections::HashMap;
use std::fs::File;
use uuid::Uuid;

use crate::asm::{AsmProg, AsmRecord};
use crate::instr::Instruction;

use lrlex::lrlex_mod;
use lrpar::lrpar_mod;

use std::io::{Read, Write};

lrlex_mod!("asm.l");
lrpar_mod!("asm.y");

pub struct Parser {
    pub isa: serde_json::Value,
    pub arch: serde_json::Value,
}

impl Parser {
    pub fn new(arch_file: &String, isa_file: &String) -> Parser {
        let mut file = File::open(isa_file).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let isa: serde_json::Value = serde_json::from_str(&contents).unwrap();

        file = File::open(arch_file).unwrap();
        contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let arch: serde_json::Value = serde_json::from_str(&contents).unwrap();

        Parser {
            isa: isa,
            arch: arch,
        }
    }

    pub fn convert_num(s: String) -> i64 {
        // remove underscore from the number
        let ss = s.trim().replace("_", "");
        let num;
        if s.starts_with("0b") {
            num = i64::from_str_radix(&ss[2..], 2).unwrap();
        } else if s.starts_with("0x") {
            num = i64::from_str_radix(&ss[2..], 16).unwrap();
        } else if s.starts_with("0o") {
            num = i64::from_str_radix(&ss[2..], 8).unwrap();
        } else {
            num = ss.parse::<i64>().unwrap();
        }
        num
    }

    pub fn find_component_kind(arch: &serde_json::Value, row: i64, col: i64, slot: i64) -> String {
        let mut component_kind = String::new();
        for c in arch["cells"].as_array().unwrap() {
            let coord = c["coordinates"].as_object().unwrap();
            if coord["row"].as_i64().unwrap() == row && coord["col"].as_i64().unwrap() == col {
                let cell = c["cell"].as_object().unwrap();
                if slot < 0 {
                    component_kind = cell["controller"]["kind"].as_str().unwrap().to_string();
                } else {
                    for rs in cell["resources_list"].as_array().unwrap() {
                        let rs_slot = rs["slot"].as_i64().unwrap();
                        let rs_size = rs["size"].as_i64().unwrap();
                        if slot >= rs_slot && slot < rs_slot + rs_size {
                            component_kind = rs["kind"].as_str().unwrap().to_string();
                            break;
                        }
                    }
                }
            }
        }
        component_kind
    }

    pub fn parse(&self, asm_file: &String) -> HashMap<String, Vec<Instruction>> {
        let lexerdef = asm_l::lexerdef();
        let asm_contents = std::fs::read_to_string(asm_file).unwrap();
        let lexer = lexerdef.lexer(&asm_contents);
        let (prog, errs) = asm_y::parse(&lexer);
        for e in errs {
            error!("{}", e.pp(&lexer, &asm_y::token_epp));
            panic!();
        }
        let mut pc_table: HashMap<String, HashMap<String, i32>> = HashMap::new();
        let mut cell_instr_counter: HashMap<String, i32> = HashMap::new();
        let prog = prog.unwrap();
        let mut x = 0;
        let mut y = 0;
        let mut label = String::new();
        for record in prog.clone().records {
            if record.name == "cell" {
                x = Parser::convert_num(record.parameters.get("x").unwrap().clone());
                y = Parser::convert_num(record.parameters.get("y").unwrap().clone());
                label = format!("{}_{}", x, y);
                if !pc_table.contains_key(&label) {
                    pc_table.insert(label.clone(), HashMap::new());
                    cell_instr_counter.insert(label.clone(), 0);
                }
            } else {
                // if record has an id that does not start with _, add it to the pc table
                if record.id != "" && !record.id.starts_with("_") {
                    let pc = cell_instr_counter.get(&label).unwrap().clone();
                    pc_table
                        .get_mut(&label)
                        .unwrap()
                        .insert(record.id.clone(), pc);
                }
                let new_pc = cell_instr_counter.get(&label).unwrap() + 1;
                cell_instr_counter.insert(label.clone(), new_pc);
            }
        }
        let mut instructions = HashMap::new();
        x = 0;
        y = 0;
        label = String::new();
        for record in prog.clone().records {
            if record.name == "cell" {
                x = Parser::convert_num(record.parameters.get("x").unwrap().clone());
                y = Parser::convert_num(record.parameters.get("y").unwrap().clone());
                label = format!("{}_{}", x, y);
                if !instructions.contains_key(&label) {
                    instructions.insert(label.clone(), Vec::new());
                    pc_table.insert(label.clone(), HashMap::new());
                }
            } else {
                let mut slot = -1;
                let mut value_map: HashMap<String, i64> = HashMap::new();
                for (key, value) in record.parameters {
                    // if value does not start with number, it is a label, we use the pc_table to resolve it
                    if !value.chars().next().unwrap().is_numeric() {
                        let pc = pc_table.get(&label).unwrap().get(&value).unwrap();
                        value_map.insert(key.clone(), *pc as i64);
                    } else {
                        value_map.insert(key.clone(), Parser::convert_num(value.clone()));
                        if key == "slot" {
                            slot = Parser::convert_num(value);
                        }
                    }
                }
                let component_kind = Parser::find_component_kind(&self.arch, x, y, slot);
                if component_kind == "" {
                    panic!("Component kind not found: x={}, y={}, slot={}", x, y, slot);
                }
                let instr =
                    Instruction::new_from_map(&record.name, &component_kind, &self.isa, &value_map);
                instructions.get_mut(&label).unwrap().push(instr);

                // if record has an id that does not start with _, add it to the pc table
                if record.id != "" && !record.id.starts_with("_") {
                    let pc = instructions.get(&label).unwrap().len() as i32 - 1;
                    pc_table
                        .get_mut(&label)
                        .unwrap()
                        .insert(record.id.clone(), pc);
                }
            }
        }

        // Now resolve the labels
        instructions
    }
}
