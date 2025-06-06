use std::collections::HashSet;
use std::fs;
use std::path::Path;

pub fn generate(arch_file_path: &Path, output_dir_path: &Path) {
    let isa_file = Path::new(&output_dir_path).join("isa.json");
    gen_isa_json(arch_file_path, &isa_file).expect("Failed to generate ISA JSON");
    let doc_file = Path::new(&output_dir_path).join("isa.md");
    gen_isa_doc(&isa_file, &doc_file).expect("Failed to generate ISA documentation");
}

fn gen_isa_doc(isa_file_path: &Path, output_doc_path: &Path) -> Result<(), std::io::Error> {
    // if the output directory does not exist, create it
    let output_dir = std::path::Path::new(output_doc_path).parent().unwrap();
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // read the json file
    let json_str = std::fs::read_to_string(isa_file_path).expect("Failed to read file");
    let isa: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

    // using minijinja to generate the ISA documentation
    let template = include_str!("../assets/isa_doc_template.md.jinja");
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_template("doc", template).unwrap();
    let tmpl = env.get_template("doc").unwrap();
    let result = tmpl.render(minijinja::context!(isa)).unwrap();

    // write the result to the output file
    match std::fs::write(output_doc_path, result) {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("Failed to write documentation to file: {}", e);
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to write documentation",
            ))
        }
    }
}

fn gen_isa_json(arch_file_path: &Path, output_file_path: &Path) -> Result<(), std::io::Error> {
    // if output directory does not exist, create it
    let output_dir = std::path::Path::new(output_file_path).parent().unwrap();
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // read the json file
    let json_str = std::fs::read_to_string(arch_file_path).expect("Failed to read file");
    let arch: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

    let mut kind_set = HashSet::new();

    let mut isa_json = serde_json::json!({
        "__comment": "This file was automatically generated by Vesyla. DO NOT EDIT.",
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
            if kind_set.is_empty() {
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
                if kind_set.is_empty() {
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
    match serde_json::to_writer_pretty(
        std::fs::File::create(output_file_path).expect("Failed to create file"),
        &isa_json,
    ) {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("Failed to write JSON to file: {}", e);
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to write JSON",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn test_isa_doc_template_included() {
        let template = include_str!("../assets/isa_doc_template.md.jinja");
        assert!(!template.is_empty(), "Template should not be empty");
        assert!(template.contains("{{")); // Check that the file contains Jinja2 syntax
    }

    #[test]
    fn test_generate_isa_doc_runs() {
        let mut tmp_isa_json = tempfile::NamedTempFile::new().expect("Failed to create temp file");
        let isa_json = include_str!("../assets/isa_example.json");
        tmp_isa_json
            .write_all(isa_json.as_bytes())
            .expect("Failed to write to temp file");
        let tmp_isa_json_path = tmp_isa_json.path();
        let dummy_isa = "ISA Example";
        let mut doc_tmp_isa_json =
            tempfile::NamedTempFile::new().expect("Failed to create temp file for doc");
        doc_tmp_isa_json
            .write_all(dummy_isa.as_bytes())
            .expect("Failed to write to temp file");

        gen_isa_doc(tmp_isa_json_path, doc_tmp_isa_json.path())
            .expect("Failed to generate ISA documentation");

        let output_content =
            std::fs::read_to_string(doc_tmp_isa_json.path()).expect("Failed to read doc file");

        assert!(
            doc_tmp_isa_json.path().exists(),
            "Documentation file should be created"
        );
        assert!(
            output_content.contains("# ISA Specification"),
            "Template header not found"
        );
        assert!(
            output_content.contains("## Instructions For Each Component"),
            "Template instructions section not found"
        );
        assert!(
            output_content.contains("### sequencer (controller)"),
            "Sequencer section not found"
        );
    }

    #[test]
    fn test_generate_isa_json_runs() {
        let tmp_arch_json = tempfile::NamedTempFile::new().expect("Failed to create temp file");
        let arch_json = include_str!("../assets/arch_example.json");
        tmp_arch_json
            .as_file()
            .write_all(arch_json.as_bytes())
            .expect("Failed to write to temp file");
        let tmp_arch_json_path = tmp_arch_json.path();

        let tmp_output_isa_json =
            tempfile::NamedTempFile::new().expect("Failed to create temp file for output");
        let tmp_output_isa_json_path = tmp_output_isa_json.path();

        gen_isa_json(tmp_arch_json_path, tmp_output_isa_json_path)
            .expect("Failed to generate ISA JSON");

        let output_content =
            std::fs::read_to_string(tmp_output_isa_json_path).expect("Failed to read output file");

        assert!(
            tmp_output_isa_json_path.exists(),
            "Output ISA JSON file should be created"
        );
        assert!(
            output_content.contains(
                "\"__comment\": \"This file was automatically generated by Vesyla. DO NOT EDIT.\""
            ),
            "Comment not found in output"
        );
        assert!(
            output_content.contains("\"format\": {"),
            "Format section not found in output"
        );
    }
}
