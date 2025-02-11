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

pub fn generate(arch_file: &String, output_dir: &String) {
    gen_bender(arch_file, output_dir);
}

fn gen_bender(arch_file: &String, output_dir: &String) {
    // if the output directory does not exist, create it
    let output_dir = std::path::Path::new(output_dir);
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // read the json file
    let json_str = std::fs::read_to_string(arch_file).expect("Failed to read file");
    let arch: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

    let mut component_set: HashSet<String> = HashSet::new();
    let mut file_list: Vec<String> = Vec::new();

    file_list.push("fabric_tb:sim.sv".to_string());
    file_list.push("fabric.sv".to_string());

    let mut component_list = Vec::new();

    for c in arch["cells"].as_array().unwrap() {
        let cell = c["cell"].as_object().unwrap();
        // check if the cell["name"] is in the set
        if !component_set.contains(cell["name"].as_str().unwrap()) {
            component_set.insert(cell["name"].as_str().unwrap().to_string());
            component_list.push(cell);
            file_list.push(format!("{}.sv", cell["name"].as_str().unwrap()));
        }
    }

    for cell in component_list {
        let controller = cell["controller"].as_object().unwrap();
        if !component_set.contains(controller["name"].as_str().unwrap()) {
            component_set.insert(controller["name"].as_str().unwrap().to_string());
            file_list.push(format!("{}.sv", controller["name"].as_str().unwrap()));
        }

        for rs in cell["resources_list"].as_array().unwrap() {
            let resource = rs.as_object().unwrap();
            if !component_set.contains(resource["name"].as_str().unwrap()) {
                component_set.insert(resource["name"].as_str().unwrap().to_string());
                file_list.push(format!("{}.sv", resource["name"].as_str().unwrap()));
            }
        }
    }

    // get environment variable VESYLA_SUITE_PATH_COMPONENTS, if variable is not set, panic
    let components_dir = std::env::var("VESYLA_SUITE_PATH_COMPONENTS")
        .expect("Environment variable VESYLA_SUITE_PATH_COMPONENTS is not set");
    let util_components = std::path::Path::new(&components_dir).join("utils");
    // get all .sv files in the utils directory
    for entry in fs::read_dir(util_components).expect("Failed to read directory") {
        let entry = entry.expect("Failed to get entry");
        let path = entry.path();
        if path.is_file() {
            let ext = path.extension().unwrap();
            if ext == "sv" {
                // get the filename including extension
                let filename = path.file_name().unwrap();
                file_list.push(filename.to_str().unwrap().to_string());
            }
        }
    }

    // reverse the order of the file list
    file_list.reverse();

    let mut file_list_with_target = Vec::new();
    for f in file_list.iter() {
        // get filename excluding extension
        let filename = f.split("/").last().unwrap().split(".").next().unwrap();
        // split filename by :
        let parts: Vec<String> = filename.split(":").map(|s| s.to_string()).collect();
        let mut targets = Vec::new();
        if parts.len() > 1 {
            // exclude the first part of the split, only save the other parts
            targets = parts[1..].iter().map(|s| s.to_string()).collect();
        } else {
            targets.push("*".to_string());
        }
        // sort the targets
        targets.sort();
        let mut targets_str = String::new();
        if targets.len() == 1 {
            targets_str = targets[0].clone();
        } else {
            // create a string: any(TARGET[0], TARGET[1], ...)
            targets_str = format!("any({})", targets.join(", "));
        }
        let mut pair = HashMap::new();
        pair.insert("target", targets_str);
        pair.insert("files", f.clone());
        file_list_with_target.push(pair);
    }

    let mut data = serde_json::json!({
        "name": "fabric",
        "dependencies": [],
        "sources": []
    });

    let mut last_targets_str = String::new();
    for f in file_list_with_target.iter() {
        let targets_str = f.get("target").unwrap().clone();
        let files_str = f.get("files").unwrap().clone();
        if targets_str != last_targets_str {
            let pair = serde_json::json!({
                "target": targets_str,
                "files": [files_str]
            });
            data["sources"].as_array_mut().unwrap().push(pair);
            last_targets_str = targets_str.to_string();
        } else {
            let pair = data["sources"].as_array_mut().unwrap().last_mut().unwrap();
            pair["files"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::Value::String(files_str));
        }
    }

    let template = r#"
package:
  name: {{ data.name }}
  authors: ["SiLago team <silago-team@eecs.kth.se>"]
{% if data.dependencies %}
dependencies:
  {%- for dep in data.dependencies %}
  {%- if dep.type == "git" %}
  {{ dep.name }}: { git: "{{ dep.git_url}}", rev: "{{ dep.git_rev }}" }
  {%- else %}
  {{ dep.name }}: { path: "{{ dep.path }}" }
  {%- endif %}
  {%- endfor %}
{% endif %}
sources:
  {%- for source in data.sources %}
  - target: {{ source.target }}
    files:
    {%- for file in source.files %}
    - {{ file }}
    {%- endfor %}
  {%- endfor %}
}"#;

    let mut env = minijinja::Environment::new();
    env.add_template("bender", template).unwrap();
    let tmpl = env.get_template("bender").unwrap();
    let result = tmpl.render(minijinja::context!(data)).unwrap();

    // write the result to the output file
    let output_file = output_dir.join("bender.yaml");
    std::fs::write(output_file, result).expect("Failed to write file");
}
