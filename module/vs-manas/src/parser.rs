#![allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use serde_json;
use std::cell;
use std::collections::HashMap;
use std::f32::consts::E;
use std::fmt::format;
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

    pub fn eval_num(s: String) -> Result<i64, String> {
        // trim all whitespaces
        let ss = s.trim();
        // check positive or negative number
        let mut sign = false;
        let mut num_str = ss;
        if ss.starts_with("-") {
            sign = true;
            num_str = &ss[1..];
        } else if ss.starts_with("+") {
            num_str = &ss[1..];
        }
        // remove underscore from the number
        let pure_num_str = num_str.replace("_", "");

        // convert the number based on the prefix, if no prefix, it is a decimal number
        let num = if pure_num_str.starts_with("0b") {
            match i64::from_str_radix(&pure_num_str[2..], 2) {
                Ok(num) => Ok(num),
                Err(_) => Err(()),
            }
        } else if pure_num_str.starts_with("0x") {
            match i64::from_str_radix(&pure_num_str[2..], 16) {
                Ok(num) => Ok(num),
                Err(_) => Err(()),
            }
        } else if pure_num_str.starts_with("0o") {
            match i64::from_str_radix(&pure_num_str[2..], 8) {
                Ok(num) => Ok(num),
                Err(_) => Err(()),
            }
        } else {
            match pure_num_str.parse::<i64>() {
                Ok(num) => Ok(num),
                Err(_) => Err(()),
            }
        };

        // if number is negative, make it negative
        match num {
            Ok(num) => {
                if sign {
                    Ok(-num)
                } else {
                    Ok(num)
                }
            }
            Err(()) => Err(format!("Invalid number: {}", s)),
        }
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
        let mut x;
        let mut y;
        let mut label = String::new();
        for record in prog.clone().records {
            if record.name == "cell" {
                x = Parser::eval_num(record.parameters.get("x").unwrap().clone()).unwrap();
                y = Parser::eval_num(record.parameters.get("y").unwrap().clone()).unwrap();
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
                x = Parser::eval_num(record.parameters.get("x").unwrap().clone()).unwrap();
                y = Parser::eval_num(record.parameters.get("y").unwrap().clone()).unwrap();
                label = format!("{}_{}", x, y);
                if !instructions.contains_key(&label) {
                    instructions.insert(label.clone(), Vec::new());
                    pc_table.insert(label.clone(), HashMap::new());
                }
            } else {
                let mut slot = -1;
                let mut value_map: HashMap<String, i64> = HashMap::new();
                for (key, value) in record.parameters {
                    let ret = Parser::eval_num(value.clone());
                    if ret.is_ok() {
                        let num = ret.unwrap();
                        value_map.insert(key.clone(), num);
                        if key == "slot" {
                            slot = num;
                        }
                    } else {
                        let pc = pc_table.get(&label).unwrap().get(&value).unwrap();
                        value_map.insert(key.clone(), *pc as i64);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_num() {
        assert_eq!(Parser::eval_num("0b1010".to_string()), Ok(10));
        assert_eq!(Parser::eval_num("0x10".to_string()), Ok(16));
        assert_eq!(Parser::eval_num("0o10".to_string()), Ok(8));
        assert_eq!(Parser::eval_num("10".to_string()), Ok(10));
        assert_eq!(Parser::eval_num("-10".to_string()), Ok(-10));
        assert_eq!(Parser::eval_num("+10".to_string()), Ok(10));
        assert_eq!(Parser::eval_num("10_000".to_string()), Ok(10000));
        assert_eq!(Parser::eval_num("0b10_000".to_string()), Ok(16));
        assert_eq!(Parser::eval_num("0x10_000".to_string()), Ok(65536));
        assert_eq!(Parser::eval_num("0o10_000".to_string()), Ok(4096));
        assert_eq!(Parser::eval_num("10_000_000".to_string()), Ok(10000000));
        assert_eq!(Parser::eval_num("0b10_000_000".to_string()), Ok(128));
        assert_eq!(Parser::eval_num("0x10_0e0_000".to_string()), Ok(269352960));
        assert_eq!(Parser::eval_num("0o10_000_071".to_string()), Ok(2097209));
        assert_eq!(Parser::eval_num("0b10_000_000_000".to_string()), Ok(1024));
        assert_eq!(
            Parser::eval_num("0b10_000_000_002".to_string()),
            Err("Invalid number: 0b10_000_000_002".to_string())
        );
    }
}
