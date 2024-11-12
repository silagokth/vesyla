#![allow(unused_imports)]

use argparse;
use log::{debug, error, info, trace, warn};
use minijinja;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::io::Result;
use std::path::PathBuf;

mod collector;
mod generator;

fn main() {
    // set the log level
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();

    // parse command line arguments with the following format:
    // program_name [command] [options]
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        error!("Usage: {} [command] [options]", args[0]);
        std::process::exit(1);
    }

    // get the command
    let command: &String = &args[1];
    let mut options: Vec<String> = Vec::new();
    for arg in args.iter().skip(2) {
        options.push(arg.clone());
    }
    match command.as_str() {
        "gen_api" => {
            info!("Generating system verilog api ...");
            gen_api(options);
            info!("Done!");
        }
        "gen_doc" => {
            info!("Generating markdown documentation ...");
            gen_doc(options);
            info!("Done!");
        }
        "assemble" => {
            info!("Assembling ...");
            assemble(options);
            info!("Done!");
        }
        "gen_rtl" => {
            match gen_rtl(options) {
                Ok(_) => info!("Done!"),
                Err(e) => error!("Error: {}", e),
            };
        }
        _ => {
            error!("Unknown command: {}", command);
            panic!();
        }
    }
}

fn gen_rtl(options: Vec<String>) -> Result<()> {
    // 1. parse the arguments to find the fabric.json input file
    let fabric_filepath = options[0].clone();
    let fabric_file = fs::File::open(fabric_filepath).unwrap();
    let fabric: serde_json::Value = serde_json::from_reader(fabric_file).unwrap();

    // 2. create a registry to store the parameters
    let mut parameters: HashMap<String, String> = HashMap::new();

    todo!();
    // 3. Add the fabric parameters to the registry
    //
    // 4. For each cell in the fabric
    //    - add parameters to the registry
    //    - for each resource in the cell
    //      - add parameters to the registry
    //      - check if RTL instance is already genreated
    //      - if not, generate the RTL instance using the template and parameters
    //      - add RTL instance to build files
    //      - pop the parameters
    //    - genreate the RTL for the cell using the template and parameters
    //    - pop the parameters
    // 5. Generate the top module for fabric using the template and parameters

    // DONE
    Ok(())
}

fn gen_api(args: Vec<String>) {
    // parse the "args" using argparse
    // -i <input file> -o <output file>
    let mut input_file = String::from("isa.json");
    let mut output_file = String::from("api.sv");
    {
        let mut ap = argparse::ArgumentParser::new();
        ap.set_description("Generate system verilog api");
        ap.refer(&mut input_file)
            .add_option(&["-i", "--input"], argparse::Store, "Input file");
        ap.refer(&mut output_file)
            .add_option(&["-o", "--output"], argparse::Store, "Output file");
        ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr())
            .unwrap();
    }

    generator::gen_api(input_file, output_file);
}

fn gen_doc(args: Vec<String>) {
    // parse the "args" using argparse
    // -i <input file> -o <output file>
    let mut input_file: String = String::from("isa.json");
    let mut output_file: String = String::from("doc.md");
    {
        let mut ap = argparse::ArgumentParser::new();
        ap.set_description("Generate system verilog api");
        ap.refer(&mut input_file)
            .add_option(&["-i", "--input"], argparse::Store, "Input file");
        ap.refer(&mut output_file)
            .add_option(&["-o", "--output"], argparse::Store, "Output file");
        ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr())
            .unwrap();
    }

    generator::gen_doc(input_file, output_file);
}

fn assemble(args: Vec<String>) {
    // parse the "args" using argparse
    // -i <input file> -o <output directory>
    let mut input_file = String::from("conf.json");
    let mut output_dir = String::from(".");
    {
        let mut ap = argparse::ArgumentParser::new();
        ap.set_description("Assemble the system");
        ap.refer(&mut input_file)
            .add_option(&["-i", "--input"], argparse::Store, "Input file");
        ap.refer(&mut output_dir).add_option(
            &["-o", "--output"],
            argparse::Store,
            "Output directory",
        );
        ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr())
            .unwrap();
    }

    let component_map = find_all_components();
    assemble_arch(
        &input_file,
        &format!("{}/arch.json", output_dir),
        &component_map,
    );
    assemble_isa(
        &format!("{}/arch.json", output_dir),
        &format!("{}/isa.json", output_dir),
        &component_map,
    );
    collector::collect_rtl(
        &format!("{}/arch.json", output_dir),
        &output_dir,
        &component_map,
    );
}

fn find_component(
    search_path: &str,
    component_map: &mut std::collections::HashMap<String, String>,
) {
    let paths = fs::read_dir(search_path).unwrap();
    for path in paths {
        let path = path.unwrap().path();
        if path.is_dir() {
            find_component(path.to_str().unwrap(), component_map);
        } else {
            if path.file_name().unwrap() == "arch.json" {
                let json_str = std::fs::read_to_string(&path).expect("Failed to read file");
                let component: serde_json::Value =
                    serde_json::from_str(&json_str).expect("Failed to parse json");
                let component_name = component["name"].as_str().unwrap();
                let dir = path.parent().unwrap();
                component_map.insert(
                    component_name.to_string(),
                    dir.to_str().unwrap().to_string(),
                );
            }
        }
    }
}

fn find_all_components() -> std::collections::HashMap<String, String> {
    let mut search_path_vec: Vec<String> = Vec::new();
    let vesyla_suite_path_share = std::env::var("VESYLA_SUITE_PATH_SHARE").expect("Environment variable VESYLA_SUITE_PATH_SHARE not set! Did you forget to source the setup script env.sh?");
    search_path_vec.push(format!("{}/components", vesyla_suite_path_share));
    search_path_vec.push(String::from("~/.vesyla-suite/components"));
    search_path_vec.push(String::from("./components"));

    // check if path exists, if yes, add to search path
    let mut existing_search_path_vec: Vec<String> = Vec::new();
    for search_path in search_path_vec.iter() {
        let path = PathBuf::from(search_path);
        if path.exists() {
            existing_search_path_vec.push(
                fs::canonicalize(search_path)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
            );
        }
    }

    assert!(
        existing_search_path_vec.len() > 0,
        "No component library found in the following location: \n{}",
        search_path_vec.join("\n")
    );

    // find all components as a hashmap, key is the component name, value is the component json object
    let mut component_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for search_path in existing_search_path_vec.iter() {
        // recursively find all directories that contains a arch.json file
        find_component(search_path, &mut component_map);
    }
    component_map
}

fn assemble_arch(
    input_file: &String,
    output_file: &String,
    component_map: &std::collections::HashMap<String, String>,
) {
    // read the json file and find all used cells
    let json_str = std::fs::read_to_string(input_file).expect("Failed to read file");
    let mut arch: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse json");

    // check mandatory fields
    assert!(
        arch["platform"].is_string(),
        "Field platform is not present or is not a string"
    );
    assert!(
        arch["instr_bitwidth"].is_number(),
        "Field instr_bitwidth is not present or is not a number"
    );
    assert!(
        arch["instr_code_bitwidth"].is_number(),
        "Field instr_code_bitwidth is not present or is not a number"
    );
    assert!(
        arch["fabric"]["cell_lists"].is_array(),
        "Field fabric.cell_lists is not present or is not an array"
    );

    let cell_lists = &arch["fabric"]["cell_lists"];
    let mut cell_name_vec: Vec<String> = Vec::new();
    for cell_list in cell_lists.as_array().unwrap() {
        let cell_list_object = cell_list.as_object().unwrap();
        let cell_name = &cell_list_object["cell_name"].as_str().unwrap();
        cell_name_vec.push(cell_name.to_string());
    }

    // create a component map content to store the component json object
    let mut component_map_content = std::collections::HashMap::new();
    for (component_name, component_dir) in component_map.iter() {
        let json_str = std::fs::read_to_string(component_dir.to_owned() + "/arch.json")
            .expect("Failed to read file");
        let component: serde_json::Value =
            serde_json::from_str(&json_str).expect("Failed to parse json");
        component_map_content.insert(component_name, component);
    }

    if arch["cells"].is_null() {
        arch["cells"] = serde_json::Value::Array(Vec::new());
    } else {
        arch["cells"].as_array_mut().unwrap().clear();
    }
    if arch["controllers"].is_null() {
        arch["controllers"] = serde_json::Value::Array(Vec::new());
    } else {
        arch["controllers"].as_array_mut().unwrap().clear();
    }
    if arch["resources"].is_null() {
        arch["resources"] = serde_json::Value::Array(Vec::new());
    } else {
        arch["resources"].as_array_mut().unwrap().clear();
    }
    for cell_name in cell_name_vec.iter() {
        if !component_map_content.contains_key(&cell_name) {
            error!("Component {} not found in the component library", cell_name);
            panic!();
        }
        // iterate through the cells, if the name matches, continue without adding it to the arch
        let mut flag_cell = false;
        for cell in arch["cells"].as_array().unwrap() {
            if cell["name"].as_str().unwrap() == cell_name {
                flag_cell = true;
            }
        }
        if !flag_cell {
            assert!(
                component_map_content[cell_name]["type"].as_str().unwrap() == "cell",
                "Component {} is not a cell",
                cell_name
            );
            arch["cells"]
                .as_array_mut()
                .unwrap()
                .push(component_map_content[cell_name].clone());
            let controller_name = component_map_content[cell_name]["controller"]
                .as_str()
                .unwrap()
                .to_string();
            if !component_map_content.contains_key(&controller_name) {
                error!(
                    "Controller {} not found in the component library",
                    controller_name
                );
                panic!();
            }
            // iterate through the controllers, if the name matches, continue without adding it to the arch
            let mut flag_controller = false;
            for controller in arch["controllers"].as_array().unwrap() {
                if controller["name"].as_str().unwrap() == controller_name {
                    flag_controller = true;
                }
            }
            if !flag_controller {
                assert!(
                    component_map_content[&controller_name]["type"]
                        .as_str()
                        .unwrap()
                        == "controller",
                    "Component {} is not a controller",
                    controller_name
                );
                arch["controllers"]
                    .as_array_mut()
                    .unwrap()
                    .push(component_map_content[&controller_name].clone());
            }
            let resource_name_vec: &Vec<serde_json::Value> = component_map_content[cell_name]
                ["resource_list"]
                .as_array()
                .unwrap();
            for resource_name in resource_name_vec.iter() {
                let resource_name: String = resource_name.as_str().unwrap().to_string();
                // iterate through the resources, if the name matches, continue without adding it to the arch
                let mut flag_resource = false;
                for resource in arch["resources"].as_array().unwrap() {
                    if resource["name"].as_str().unwrap() == resource_name {
                        flag_resource = true;
                    }
                }
                if !flag_resource {
                    if !component_map_content.contains_key(&resource_name) {
                        error!(
                            "Resource {} not found in the component library",
                            resource_name
                        );
                        panic!();
                    }
                    assert!(
                        component_map_content[&resource_name]["type"]
                            .as_str()
                            .unwrap()
                            == "resource",
                        "Component {} is not a resource",
                        resource_name
                    );
                    arch["resources"]
                        .as_array_mut()
                        .unwrap()
                        .push(component_map_content[&resource_name].clone());
                }
            }
        }
    }

    // write the result to the output file
    std::fs::write(output_file, serde_json::to_string_pretty(&arch).unwrap())
        .expect("Failed to write file");
}

fn assemble_isa(
    arch_file: &String,
    output_file: &String,
    component_map: &std::collections::HashMap<String, String>,
) {
    // read the json file and find all used cells
    let json_str = std::fs::read_to_string(arch_file).expect("Failed to read file");
    let arch: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

    let mut controller_list: Vec<String> = Vec::new();
    let mut resource_list: Vec<String> = Vec::new();

    for controller in arch["controllers"].as_array().unwrap() {
        let controller_name = controller["name"].as_str().unwrap();
        controller_list.push(controller_name.to_string());
    }
    for resource in arch["resources"].as_array().unwrap() {
        let resource_name = resource["name"].as_str().unwrap();
        resource_list.push(resource_name.to_string());
    }

    let mut isa = serde_json::json!({
        "platform": arch["platform"],
        "instr_bitwidth": arch["instr_bitwidth"],
        "instr_code_bitwidth": arch["instr_code_bitwidth"],
        "components": []
    });

    for component in controller_list {
        let mut js = serde_json::json!({
            "name": component,
            "type": "controller",
            "instruction_templates": []
        });
        let json_str = std::fs::read_to_string(format!("{}/isa.json", component_map[&component]))
            .expect("Failed to read file");
        let json_isa: serde_json::Value =
            serde_json::from_str(&json_str).expect("Failed to parse json");
        js["instruction_templates"] = json_isa;
        isa["components"].as_array_mut().unwrap().push(js);
    }

    for component in resource_list {
        let mut js = serde_json::json!({
            "name": component,
            "type": "resource",
            "instruction_templates": []
        });
        let json_str = std::fs::read_to_string(format!("{}/isa.json", component_map[&component]))
            .expect("Failed to read file");
        let json_isa: serde_json::Value =
            serde_json::from_str(&json_str).expect("Failed to parse json");
        js["instruction_templates"] = json_isa;
        isa["components"].as_array_mut().unwrap().push(js);
    }

    // write the result to the output file
    std::fs::write(output_file, serde_json::to_string_pretty(&isa).unwrap())
        .expect("Failed to write file");
}
