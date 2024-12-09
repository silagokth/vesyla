use argparse;
use chrono::Local;
use glob::glob;
use log::{debug, error, info, trace, warn};
use std::env;
use std::fs::{self, File};
use std::io;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process;

struct DataBase {
    isa: InstructionSet,
    pkg: InstructionPackage,
}

struct InstructionSet {
    platform: string,
    format: map<string, int>,
    instruction_templates: map<string, Vec<InstructionTemplate>>,
}

struct InstructionTemplate {
    name: string,
    opcode: int,
    instr_type: string,
    segment_templates: Vec<SegmentTemplate>,
}

struct SegmentTemplate {
    name: string,
    bitwidth: int,
    is_signed: bool,
    default_val: int,
    verbose_map: Map<int, string>,
    comment: string,
}

struct InstructionPackage {
    coord: Map<string, int>,
    instructions: Vec<Instruction>,
}

struct Instruction {
    name: string,
    value_map: map<string, int>,
    int_info_map: map<string, int>,
    str_info_map: map<string, string>,
}

impl Instruction {
    fn new() -> Instruction {
        Instruction {
            name: "".to_string(),
            value_map: map<string, int>::new(),
            int_info_map: map<string, int>::new(),
            str_info_map: map<string, string>::new(),
        }
    }
    fn from_json(json_str : &str) -> Instruction {
        let j: Value = serde_json::from_str(json_str).unwrap();
        let mut instr = Instruction::new();
        instr.name = j["name"].as_str().unwrap().to_string();
        let value_map = j["value_map"].as_object().unwrap();
        for (k, v) in value_map {
            instr.value_map.insert(k.to_string(), v.as_i64().unwrap() as int);
        }
        let int_info_map = j["int_info_map"].as_object().unwrap();
        for (k, v) in int_info_map {
            instr.int_info_map.insert(k.to_string(), v.as_i64().unwrap() as int);
        }
        let str_info_map = j["str_info_map"].as_object().unwrap();
        for (k, v) in str_info_map {
            instr.str_info_map.insert(k.to_string(), v.as_str().unwrap().to_string());
        }
        instr
    }
    fn to_json(&self) -> string {
        let mut j = json::object! {
            "name" => self.name,
            "value_map" => json::object! {},
            "int_info_map" => json::object! {},
            "str_info_map" => json::object! {},
        };
        for (k, v) in &self.value_map {
            j["value_map"][k] = v.into();
        }
        for (k, v) in &self.int_info_map {
            j["int_info_map"][k] = v.into();
        }
        for (k, v) in &self.str_info_map {
            j["str_info_map"][k] = v.into();
        }
        j.dump()
    }
}

struct ASMProg {
    contents: Vec<String>,
    records: map<string, ASMRecord>,
}

struct ASMRecord {
    id: string,
    kind: Kind,
    name: string,
    parameters: map<string, string>,
}

enum Kind {
    UNKNOWN,
    CELL,
    INSTR,
}
