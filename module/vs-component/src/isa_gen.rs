#![allow(unused_imports)]

use jsonschema::output;
use log::{debug, error, info, trace, warn};
use minijinja;
use serde_json;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::hash::Hash;
use std::string;

pub fn gen_isa_doc(isa_file: &String, doc_file: &String) {
    // if the output directory does not exist, create it
    let output_dir = std::path::Path::new(doc_file).parent().unwrap();
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // read the json file
    let json_str = std::fs::read_to_string(isa_file).expect("Failed to read file");
    let isa: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

    // using minijinja to generate the system verilog api
    let template = r#"

# ISA Specification

Instructions are {{isa.format.instr_bitwidth}}-bit wide. The MSB indicates whether it's a control instruction or a resource instruction. [0]: control; [1]: resource; The next {{instr_opcode_bitwidth}} bits represent the instruction opcode. The rest of the bits are used to encode the instruction content. For resource instructions, another {{instr_slot_bitwidth}} bits in the instruction content are used to indicate the slot number. The rest of the bits are used to encode the instruction content.

Note that, specifically for resource instructions, if instruction opcode start with "11", the instruction contains a field that need to be replaced by RACCU registers if the filed is marked "dynamic".

## ISA Format
Parameter | Width | Description
----------|-------|-------------------------
instr_bitwidth | {{isa.format.instr_bitwidth}} | Instruction bitwidth
instr_type_bitwidth | {{isa.format.instr_type_bitwidth}} | Instruction type bitwidth
instr_opcode_bitwidth | {{isa.format.instr_opcode_bitwidth}} | Instruction opcode bitwidth
instr_slot_bitwidth | {{isa.format.instr_slot_bitwidth}} | Instruction slot bitwidth, only used for resource components

## Instructions For Each Component
{%- for c in isa.components %}

### {{c.kind}} ( {{c.component_type}} )

{% if c.component_type=="controller" %}
{%- set useful_bitwidth=isa.format.instr_bitwidth - isa.format.instr_opcode_bitwidth -%}
{% else %}
{%- set useful_bitwidth=isa.format.instr_bitwidth - isa.format.instr_opcode_bitwidth - isa.format.instr_slot_bitwidth -%}
{% endif %}

{%- for i in c.instructions %}

#### {{i.name}} [opcode={{i.opcode}}]

Field | Position | Width | Default Value | Description
------|----------|-------|---------------|-------------------------
{%- set start =useful_bitwidth-1 %}
{%- for j in i.segments %}
{%- set end = start - j.bitwidth + 1 %}
{{j.name}} | [{{start }}, {{end}}] | {{j.bitwidth}} | {%- if j.default_val is defined -%}{{j.default_val}}{%- else -%}0{%- endif -%} | {{j.comment}} {%- if j.verbo_map -%}{% for k in j.verbo_map %} [{{k.key}}]:{{k.val}};{% endfor %}{%- endif -%}
{%- set start = end - 1 -%}
{%- endfor -%}
{%- endfor -%}

{%- endfor -%}

"#;

    let mut env = minijinja::Environment::new();
    env.add_template("doc", template).unwrap();
    let tmpl = env.get_template("doc").unwrap();
    let result = tmpl.render(minijinja::context!(isa)).unwrap();

    // write the result to the output file
    std::fs::write(doc_file, result).expect("Failed to write file");
}

pub fn gen_isa_json(arch_file: &String, isa_file: &String) {
    // if output directory does not exist, create it
    let output_dir = std::path::Path::new(isa_file).parent().unwrap();
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // read the json file
    let json_str = std::fs::read_to_string(arch_file).expect("Failed to read file");
    let arch: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

    let mut kind_set = HashSet::new();

    let mut isa_json = serde_json::json!({
        "format": {
            "instr_bitwidth": 0,
            "instr_type_bitwidth": 0,
            "instr_opcode_bitwidth": 0,
            "instr_slot_bitwidth": 0
        },
        "components": []
    });

    for c in arch["cells"].as_array().unwrap() {
        let cell = c["cell"].as_object().unwrap();
        let controller = cell["controller"].as_object().unwrap();
        // check if the controller kind is in the set
        if !kind_set.contains(controller["kind"].as_str().unwrap()) {
            if kind_set.len() == 0 {
                isa_json["format"]["instr_bitwidth"] =
                    controller["isa"]["format"]["instr_bitwidth"].clone();
                isa_json["format"]["instr_type_bitwidth"] =
                    controller["isa"]["format"]["instr_type_bitwidth"].clone();
                isa_json["format"]["instr_opcode_bitwidth"] =
                    controller["isa"]["format"]["instr_opcode_bitwidth"].clone();
                isa_json["format"]["instr_slot_bitwidth"] =
                    controller["isa"]["format"]["instr_slot_bitwidth"].clone();
            } else {
                if isa_json["format"]["instr_bitwidth"].as_u64().unwrap()
                    != controller["isa"]["format"]["instr_bitwidth"]
                        .as_u64()
                        .unwrap()
                {
                    panic!("Instruction bitwidth mismatch");
                }
                if isa_json["format"]["instr_type_bitwidth"].as_u64().unwrap()
                    != controller["isa"]["format"]["instr_type_bitwidth"]
                        .as_u64()
                        .unwrap()
                {
                    panic!("Type bitwidth mismatch");
                }
                if isa_json["format"]["instr_opcode_bitwidth"]
                    .as_u64()
                    .unwrap()
                    != controller["isa"]["format"]["instr_opcode_bitwidth"]
                        .as_u64()
                        .unwrap()
                {
                    panic!("Opcode bitwidth mismatch");
                }
                if isa_json["format"]["instr_slot_bitwidth"].as_u64().unwrap()
                    != controller["isa"]["format"]["instr_slot_bitwidth"]
                        .as_u64()
                        .unwrap()
                {
                    panic!("Slot bitwidth mismatch");
                }
            }
            let component_obj = serde_json::json!({
                "kind": controller["kind"].clone(),
                "component_type" : controller["component_type"].clone(),
                "instructions": controller["isa"]["instructions"].clone()
            });
            isa_json["components"]
                .as_array_mut()
                .unwrap()
                .push(component_obj);

            kind_set.insert(controller["kind"].as_str().unwrap());
        }

        for resource in cell["resources_list"].as_array().unwrap() {
            if !kind_set.contains(resource["kind"].as_str().unwrap()) {
                if kind_set.len() == 0 {
                    isa_json["format"]["instr_bitwidth"] =
                        resource["isa"]["format"]["instr_bitwidth"].clone();
                    isa_json["format"]["instr_type_bitwidth"] =
                        resource["isa"]["format"]["instr_type_bitwidth"].clone();
                    isa_json["format"]["instr_opcode_bitwidth"] =
                        resource["isa"]["format"]["instr_opcode_bitwidth"].clone();
                    isa_json["format"]["instr_slot_bitwidth"] =
                        resource["isa"]["format"]["instr_slot_bitwidth"].clone();
                } else {
                    if isa_json["format"]["instr_bitwidth"].as_u64().unwrap()
                        != resource["isa"]["format"]["instr_bitwidth"]
                            .as_u64()
                            .unwrap()
                    {
                        panic!("Instruction bitwidth mismatch");
                    }
                    if isa_json["format"]["instr_type_bitwidth"].as_u64().unwrap()
                        != resource["isa"]["format"]["instr_type_bitwidth"]
                            .as_u64()
                            .unwrap()
                    {
                        panic!("Type bitwidth mismatch");
                    }
                    if isa_json["format"]["instr_opcode_bitwidth"]
                        .as_u64()
                        .unwrap()
                        != resource["isa"]["format"]["instr_opcode_bitwidth"]
                            .as_u64()
                            .unwrap()
                    {
                        panic!("Opcode bitwidth mismatch");
                    }
                    if isa_json["format"]["instr_slot_bitwidth"].as_u64().unwrap()
                        != resource["isa"]["format"]["instr_slot_bitwidth"]
                            .as_u64()
                            .unwrap()
                    {
                        panic!("Slot bitwidth mismatch");
                    }
                }
                let component_obj = serde_json::json!({
                    "kind": resource["kind"].clone(),
                    "component_type" : resource["component_type"].clone(),
                    "instructions": resource["isa"]["instructions"].clone()
                });
                isa_json["components"]
                    .as_array_mut()
                    .unwrap()
                    .push(component_obj);

                kind_set.insert(resource["kind"].as_str().unwrap());
            }
        }
    }

    // write the result to the output file
    std::fs::write(isa_file, isa_json.to_string()).expect("Failed to write file");
}
