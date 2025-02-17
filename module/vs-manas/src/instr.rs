use serde_json;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Instruction {
    pub name: String,
    pub kind: String,
    pub component_type: String,
    pub format: serde_json::Value,
    pub template: serde_json::Value,
    pub value_map: HashMap<String, i64>,
}

impl Instruction {
    pub fn new_from_map(
        name: &String,
        kind: &String,
        isa: &serde_json::Value,
        value_map: &HashMap<String, i64>,
    ) -> Instruction {
        let format = isa["format"].clone();
        let mut component_type = String::new();
        let mut template = serde_json::Value::Null;
        for component in isa["components"].as_array().unwrap() {
            if *kind == component["kind"].as_str().unwrap().to_string() {
                component_type = component["component_type"].as_str().unwrap().to_string();
                for instruction in component["instructions"].as_array().unwrap() {
                    if *name == instruction["name"].as_str().unwrap().to_string() {
                        template = instruction.clone();
                        break;
                    }
                }
                break;
            }
        }

        if format == serde_json::Value::Null {
            panic!("Format not found");
        }
        if template == serde_json::Value::Null {
            panic!("Instruction not found: {}", name);
        }

        Instruction {
            name: name.clone(),
            kind: kind.clone(),
            component_type: component_type,
            format: format,
            template: template,
            value_map: value_map.clone(),
        }
    }

    pub fn new_from_bin(
        name: &String,
        kind: &String,
        isa: &serde_json::Value,
        bin: &String,
    ) -> Instruction {
        let format = isa["format"].clone();
        let mut component_type = String::new();
        let mut template = serde_json::Value::Null;
        for component in isa["components"].as_array().unwrap() {
            if *kind == component["kind"].as_str().unwrap().to_string() {
                component_type = component["component_type"].as_str().unwrap().to_string();
                for instruction in component["instructions"].as_array().unwrap() {
                    if *name == instruction["name"].as_str().unwrap().to_string() {
                        template = instruction.clone();
                        break;
                    }
                }
                break;
            }
        }

        if format == serde_json::Value::Null {
            panic!("Format not found");
        }
        if template == serde_json::Value::Null {
            panic!("Instruction not found: {}", name);
        }

        let mut value_map = HashMap::new();
        let mut tmp = bin.clone();
        // check if bin bitwidth is equal to instr_bitwidth and if it only has 1 or 0
        if tmp.len() != format["instr_bitwidth"].as_u64().unwrap() as usize {
            panic!("Invalid instruction bitwidth");
        }
        for c in tmp.chars() {
            if c != '0' && c != '1' {
                panic!("Invalid binary string");
            }
        }
        // reverse the string to make it easier to extract fields
        tmp = tmp.chars().rev().collect::<String>();
        let instr_type_bitwidth = format["instr_type_bitwidth"].as_u64().unwrap();
        let instr_type = tmp[0..instr_type_bitwidth as usize]
            .chars()
            .rev()
            .collect::<String>()
            .parse::<i64>()
            .unwrap();
        match instr_type {
            0 => {
                if component_type != "controller" {
                    panic!("component type mismatch");
                }
            }
            1 => {
                if component_type != "resource" {
                    panic!("component type mismatch");
                }
            }
            _ => panic!("Invalid instruction type"),
        }
        tmp = tmp[instr_type_bitwidth as usize..].to_string();

        let instr_opcode_bitwidth = format["instr_opcode_bitwidth"].as_u64().unwrap();
        let instr_opcode = tmp[0..instr_opcode_bitwidth as usize]
            .chars()
            .rev()
            .collect::<String>()
            .parse::<i64>()
            .unwrap();
        if instr_opcode != template["opcode"].as_i64().unwrap() {
            panic!("opcode mismatch");
        }
        tmp = tmp[instr_opcode_bitwidth as usize..].to_string();

        if component_type == "resource" {
            let instr_slot_bitwidth = format["instr_slot_bitwidth"].as_u64().unwrap();
            let instr_slot = tmp[0..instr_slot_bitwidth as usize]
                .chars()
                .rev()
                .collect::<String>()
                .parse::<i64>()
                .unwrap();
            value_map.insert("slot".to_string(), instr_slot);
            tmp = tmp[instr_slot_bitwidth as usize..].to_string();
        }

        for field in template["fields"].as_array().unwrap() {
            let field_name = field["name"].as_str().unwrap().to_string();
            let field_bitwidth = field["bitwidth"].as_u64().unwrap();
            let field_value = tmp[0..field_bitwidth as usize]
                .chars()
                .rev()
                .collect::<String>()
                .parse::<i64>()
                .unwrap();
            value_map.insert(field_name, field_value);
            tmp = tmp[field_bitwidth as usize..].to_string();
        }

        Instruction {
            name: name.clone(),
            kind: kind.clone(),
            component_type: component_type,
            format: format,
            template: template,
            value_map: value_map,
        }
    }

    pub fn to_bin(&self) -> String {
        let mut bin = String::new();
        let instr_type_str = self.component_type.clone();
        let instr_type = match instr_type_str.as_str() {
            "controller" => format!(
                "{:0width$b}",
                0,
                width = self.format["instr_type_bitwidth"].as_u64().unwrap() as usize
            ),
            "resource" => format!(
                "{:0width$b}",
                1,
                width = self.format["instr_type_bitwidth"].as_u64().unwrap() as usize
            ),
            _ => panic!("Unsupported instruction type"),
        };
        bin = bin + &instr_type;

        let instr_opcode = format!(
            "{:0width$b}",
            self.template["opcode"].as_u64().unwrap(),
            width = self.format["instr_opcode_bitwidth"].as_u64().unwrap() as usize
        );
        bin = bin + &instr_opcode;
        if self.component_type == "resource" {
            let mut slot = 0;
            if let Some(value) = self.value_map.get("slot") {
                slot = *value;
            }
            let instr_slot = format!(
                "{:0width$b}",
                slot,
                width = self.format["instr_slot_bitwidth"].as_u64().unwrap() as usize
            );
            bin = bin + &instr_slot;
        }
        if self.template["segments"] != serde_json::Value::Null {
            for field in self.template["segments"].as_array().unwrap() {
                let field_name = field["name"].as_str().unwrap().to_string();
                let mut field_value = 0;
                if let Some(default_value) = field.get("default_value") {
                    field_value = default_value.as_i64().unwrap();
                }
                if let Some(value) = self.value_map.get(&field_name) {
                    field_value = *value;
                }
                let mut is_signed = false;
                if let Some(value) = field.get("is_signed") {
                    is_signed = value.as_bool().unwrap();
                }
                if is_signed {
                    let mask = (1 << field["bitwidth"].as_u64().unwrap()) - 1;
                    field_value = field_value & mask;
                }
                let field_bin = format!(
                    "{:0width$b}",
                    field_value,
                    width = field["bitwidth"].as_u64().unwrap() as usize
                );
                bin = bin + &field_bin;
            }
        }
        // fill the remaining bits with 0
        let instr_bitwidth = self.format["instr_bitwidth"].as_u64().unwrap();
        let bin_len = bin.len() as u64;
        if let Some(remaining_bits) = instr_bitwidth.checked_sub(bin_len) {
            for _ in 0..remaining_bits {
                bin = bin + "0";
            }
        } else {
            // Handle the case where bin_len is greater than instr_bitwidth
            // For example, you could log an error or panic
            panic!("bin length is greater than instruction bitwidth");
        }
        bin
    }

    pub fn to_txt(&self) -> String {
        let mut txt = String::new();
        txt = txt + &self.name;
        txt = txt + " (";
        let mut flag_first = true;
        // if it's a resource instruction, print the slot
        if self.component_type == "resource" {
            let mut slot = 0;
            if let Some(value) = self.value_map.get("slot") {
                slot = *value;
            }
            txt = txt + "slot=";
            txt = txt + &slot.to_string();
            flag_first = false;
        }
        if self.template["segments"] != serde_json::Value::Null {
            for field in self.template["segments"].as_array().unwrap() {
                if !flag_first {
                    txt = txt + ", ";
                }
                flag_first = false;
                let field_name = field["name"].as_str().unwrap().to_string();
                let mut field_value = 0;
                if let Some(default_value) = field.get("default_value") {
                    field_value = default_value.as_i64().unwrap();
                }
                if let Some(value) = self.value_map.get(&field_name) {
                    field_value = *value;
                }
                txt = txt + &field_name;
                txt = txt + "=";
                txt = txt + &field_value.to_string();
            }
        }
        txt = txt + " )";
        txt
    }
}
