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
    gen_file_list(arch_file, output_dir);
    gen_scripts(arch_file, output_dir);
}

fn gen_file_list(arch_file: &String, output_dir: &String) {
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

    file_list.push("fabric_tb.sv".to_string());
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
    // get all *.sv files in the util directory
    let util_files = fs::read_dir(util_components).unwrap();
    for file in util_files {
        let file = file.unwrap();
        let path = file.path();
        let ext = path.extension().unwrap();
        if ext == "sv" {
            file_list.push(path.file_name().unwrap().to_str().unwrap().to_string());
        }
    }

    // reverse the order of the file list
    file_list.reverse();

    println!("{:?}", file_list);

    // output the file list to output_dir/file_list.txt
    let file_list_path = output_dir.join("file_list.txt");
    let mut file_list_str = String::new();
    for file in file_list {
        file_list_str.push_str(&file);
        file_list_str.push_str("\n");
    }

    println!("{}", file_list_str);
    println!("{}", file_list_path.display());
    fs::write(file_list_path, file_list_str).expect("Failed to write file");
}

fn gen_scripts(arch_file: &String, output_dir: &String) {}
