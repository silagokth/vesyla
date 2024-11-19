#![allow(unused_imports)]
mod collector;
mod drra;
mod generator;

use crate::drra::{Cell, Controller, Fabric, ParameterList, Resource};
use bs58::encode;
use clap::{Parser, Subcommand, ValueEnum};
use log::{debug, error, info, trace, warn};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::env;
use std::fmt::write;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

#[derive(Subcommand)]
enum Command {
    #[command(about = "Generate system verilog api", name = "gen_api")]
    GenApi {
        #[arg(short, long)]
        input_file: String,
        #[arg(short, long)]
        output_file: String,
    },
    #[command(about = "Generate markdown documentation", name = "gen_doc")]
    GenDoc {
        #[arg(short, long)]
        input_file: String,
        #[arg(short, long)]
        output_file: String,
    },
    #[command(about = "Assemble the system", name = "assemble")]
    Assemble {
        #[arg(short, long)]
        input_file: String,
        #[arg(short, long)]
        output_dir: String,
    },
    #[command(about = "Generate RTL", name = "gen_rtl")]
    GenRtl {
        /// Path to the fabric.json file
        #[arg(short, long)]
        fabric_description: String,
        /// Debug mode (default: false)
        #[arg(short, long, default_value_t = false)]
        debug: bool,
        /// Build directory
        #[arg(short, long, default_value = "build")]
        build_dir: String,
    },
    #[command(about = "Validate JSON file", name = "validate_json")]
    ValidateJson {
        /// Path to the JSON file
        #[arg(short, long)]
        json_file: String,
        #[arg(short, long)]
        schema_file: String,
    },
}

#[derive(Parser)]
#[command(version, about, long_about=None)]
struct Args {
    /// Command to execute
    #[command(subcommand)]
    command: Command,
}

fn main() {
    // set the log level
    //env_logger::Builder::from_default_env()
    //    .filter_level(log::LevelFilter::Debug)
    //    .init();

    let cli_args = Args::parse();

    match &cli_args.command {
        Command::GenApi {
            input_file,
            output_file,
        } => {
            info!("Generating system verilog api ...");
            gen_api(input_file.clone(), output_file.clone());
            info!("Done!");
        }
        Command::GenDoc {
            input_file,
            output_file,
        } => {
            info!("Generating markdown documentation ...");
            gen_doc(input_file.clone(), output_file.clone());
            info!("Done!");
        }
        Command::Assemble {
            input_file,
            output_dir,
        } => {
            info!("Assembling ...");
            assemble(input_file.clone(), output_dir.clone());
            info!("Done!");
        }
        Command::GenRtl {
            fabric_description,
            build_dir,
            debug,
        } => {
            info!("Generating RTL...");
            if *debug {
                env_logger::Builder::from_default_env()
                    .filter_level(log::LevelFilter::Debug)
                    .init();
            } else {
                env_logger::Builder::from_default_env()
                    .filter_level(log::LevelFilter::Info)
                    .init();
            }
            match gen_rtl(fabric_description.clone(), build_dir.clone()) {
                Ok(_) => info!("Done!"),
                Err(e) => error!("Error: {}", e),
            };
        }
        Command::ValidateJson {
            json_file,
            schema_file,
        } => {
            info!("Validating JSON file ...");
            match validate_json(json_file.clone(), schema_file.clone()) {
                Ok(_) => info!("Done!"),
                Err(e) => error!("Error: {}", e),
            };
        }
    }
}

fn validate_json(json_file: String, schema_file: String) -> Result<()> {
    let json_file = match fs::File::open(Path::new(&json_file)) {
        Ok(file) => serde_json::from_reader(file).expect("Failed to parse JSON file"),
        Err(err) => {
            println!("Error: {}", err);
            panic!("Failed to open file: {}", json_file);
        }
    };
    let schema_file = match fs::File::open(Path::new(&schema_file)) {
        Ok(file) => serde_json::from_reader(file).expect("Failed to parse schema file"),
        Err(err) => {
            println!("Error: {}", err);
            panic!("Failed to open file: {}", schema_file);
        }
    };
    let validator = jsonschema::validator_for(&schema_file).expect("Failed to create validator");
    if validator.is_valid(&json_file) {
        info!("JSON file is valid");
        Ok(())
    } else {
        panic!("JSON file is not valid");
    }
}

fn gen_rtl(fabric_filepath: String, build_dir: String) -> Result<()> {
    debug!("Library path: {}", get_library_path());

    // Create lists for implemented cells, resources and controllers
    let mut implemented_cells: HashMap<String, Cell> = HashMap::new();
    let mut implemented_resources: HashMap<String, Resource> = HashMap::new();
    let mut implemented_controllers: HashMap<String, Controller> = HashMap::new();

    // Create JSON arrays to store the output cells, resources and controllers
    let mut output_cells: Vec<serde_json::Value> = Vec::new();
    let mut output_resources: Vec<serde_json::Value> = Vec::new();
    let mut output_controllers: Vec<serde_json::Value> = Vec::new();
    let mut output_cells_list: Vec<serde_json::Value> = Vec::new();

    // parse the arguments to find the fabric.json input file
    let fabric_file = match fs::File::open(Path::new(&fabric_filepath)) {
        Ok(file) => file,
        Err(err) => {
            println!("Error: {}", err);
            panic!("Failed to open file: {}", fabric_filepath);
        }
    };
    let fabric_json: serde_json::Value = serde_json::from_reader(fabric_file).unwrap_or_else(|e| {
        panic!("Failed to parse fabric.json: {}", e);
    });

    // create a registry to store the parameters
    let mut parameter_list = ParameterList::new();

    // get cell_pool, resource_pool and controller_pool from the json file
    let cell_pool = fabric_json
        .get("cells")
        .expect("Cell pool not found in fabric");
    let resource_pool = fabric_json
        .get("resources")
        .expect("Resource pool not found in fabric");
    let controller_pool = fabric_json
        .get("controllers")
        .expect("Controller pool not found in fabric");

    // get the fabric object from the json file
    let fabric = fabric_json
        .get("fabric")
        .expect("Fabric not found in .json");

    // Get fabric dimensions
    let fabric_height = fabric["height"].as_u64().unwrap();
    let fabric_width = fabric["width"].as_u64().unwrap();

    // add fabric parameters to the registry
    let fabric_parameters = get_parameters(fabric, Some("custom_properties".to_string()));
    if fabric_parameters.is_empty() {
        warn!("No fabric parameters found in {}", fabric_filepath);
    } else {
        match add_parameters(&fabric_parameters, &mut parameter_list) {
            Ok(_) => (),
            Err(e) => {
                panic!("Error with fabric.json parameters: ({})", e);
            }
        }
    }

    // Create the fabric object
    let mut fabric_object = Fabric::new(fabric_height, fabric_width);
    fabric_object.parameters = fabric_parameters;

    // CELLS
    let cell_list = fabric.get("cells_list").expect("Cells not found in fabric");
    for cell in cell_list.as_array().unwrap() {
        let mut resource_hashes = Vec::new();
        let mut cell_object = match Cell::from_json(&cell.to_string()) {
            Ok(cell) => cell,
            Err(drra::Error::ComponentWithoutNameOrKind) => {
                panic!("Cell without name or kind found in fabric description");
            }
            Err(e) => {
                panic!("Error with cell: {}", e);
            }
        };

        // Check cell coordinates
        if cell_object.coordinates_list.is_empty() {
            panic!(
                "Cell {} was declared without coordinates in fabric description",
                cell_object.name
            );
        }

        // Get cell parameters from fabric, or cell pool, or library
        let mut overwritten_params = Vec::new();

        // Get cell from cell pool
        if let Some(cell_pool_entry) = cell_pool
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| *entry["name"].as_str().unwrap() == cell_object.name)
        {
            if let Ok(cell_from_pool) = Cell::from_json(&cell_pool_entry.to_string()) {
                // If cell controller was not provided in "fabric" get from cell_from_pool
                if cell_from_pool.controller.is_some() && cell_object.controller.is_none() {
                    cell_object.controller = cell_from_pool.controller.clone();
                }
                // If cell resources were not provided in "fabric" get from cell_from_pool
                if cell_from_pool.resources.is_some() && cell_object.resources.is_none() {
                    cell_object.resources = cell_from_pool.resources.clone();
                }
                // If cell kind was not provided in "fabric" get from cell_from_pool
                if cell_from_pool.kind.is_some() && cell_object.kind.is_none() {
                    cell_object.kind = cell_from_pool.kind.clone();
                }
                // append parameters from cell_from_pool to cell_parameters
                overwritten_params.extend(merge_parameters(
                    &mut cell_object.parameters,
                    &cell_from_pool.parameters,
                )?);
                // append required parameters from cell_from_pool to cell
                if !cell_from_pool.required_parameters.is_empty() {
                    cell_object
                        .required_parameters
                        .extend(cell_from_pool.required_parameters);
                }
            }
        }

        // Get cell from library if kind is provided
        if let Some(cell_type) = cell_object.kind.as_ref() {
            if let Ok(lib_cell) = get_from_library(cell_type) {
                if let Ok(cell_from_lib) = Cell::from_json(&lib_cell.to_string()) {
                    // If cell controller was not provided in "fabric" or cell pool, get from cell_from_lib
                    if cell_from_lib.controller.is_some() && cell_object.controller.is_none() {
                        cell_object.controller = cell_from_lib.controller.clone();
                    }
                    // If cell resources were not provided in "fabric" or cell pool, get from cell_from_lib
                    if cell_from_lib.resources.is_some() && cell_object.resources.is_none() {
                        cell_object.resources = cell_from_lib.resources.clone();
                    }
                    // append parameters from cell_from_lib to cell_parameters
                    overwritten_params.extend(merge_parameters(
                        &mut cell_object.parameters,
                        &cell_from_lib.parameters,
                    )?);
                    if !overwritten_params.is_empty() {
                        let mut warning = format!(
                            "Some parameters from cell {} were overwritten:",
                            cell_object.name,
                        );
                        for param in overwritten_params.iter() {
                            warning.push_str(&format!(
                                "\n - {}(old value: {}, new value: {})",
                                param.0, param.1, param.2
                            ));
                        }
                        debug!("{}", warning);
                    }
                    // append required parameters from cell_from_lib to cell
                    if !cell_from_lib.required_parameters.is_empty() {
                        cell_object
                            .required_parameters
                            .extend(cell_from_lib.required_parameters);
                    }
                }
            }
        }

        // Verify cell validity
        if cell_object.controller.is_none() {
            panic!("Controller not found for cell {}", cell_object.name,);
        }
        if cell_object.resources.is_none() {
            panic!("Resources for cell {} not found in the library and were not provided in JSON file fabric or cell pool", cell_object.name);
        }
        if cell_object.parameters.is_empty() && cell_object.required_parameters.is_empty() {
            warn!(
                "No parameters or required_parameters found for cell {}",
                cell_object.name,
            );
        } else {
            // Add cell parameters to the registry
            match add_parameters(&cell_object.parameters, &mut parameter_list) {
                Ok(_) => (),
                Err(e) => {
                    panic!(
                        "Error with cell parameters for cell ({}): {}",
                        cell_object.name, e,
                    );
                }
            }
        }

        // CONTROLLER
        let mut overwritten_params = Vec::new();
        // Get the controller from the controller pool
        if cell_object.controller.as_mut().unwrap().size.is_none() {
            if let Some(controller) = controller_pool.as_array().unwrap().iter().find(|entry| {
                entry["name"].as_str().unwrap() == cell_object.controller.as_ref().unwrap().name
            }) {
                if let Ok(controller_from_pool) = Controller::from_json(&controller.to_string()) {
                    // Check size
                    cell_object.controller.as_mut().unwrap().size = controller_from_pool.size;
                    // append parameters from controller_from_pool to controller_parameters
                    overwritten_params.extend(merge_parameters(
                        &mut cell_object.controller.as_mut().unwrap().parameters,
                        &controller_from_pool.parameters,
                    )?);
                    // append required parameters from controller_from_pool to controller
                    if !controller_from_pool.required_parameters.is_empty() {
                        cell_object
                            .controller
                            .as_mut()
                            .unwrap()
                            .required_parameters
                            .extend(controller_from_pool.required_parameters);
                    }
                    // check if the controller has kind
                    if controller_from_pool.kind.is_some()
                        && cell_object.controller.as_ref().unwrap().kind.is_none()
                    {
                        cell_object.controller.as_mut().unwrap().kind = controller_from_pool.kind;
                    }
                }
            }
        }

        // Check if the controller has kind
        if cell_object.controller.as_ref().unwrap().kind.is_none() {
            panic!(
                "Kind not found for controller {} in cell {}",
                cell_object.controller.as_ref().unwrap().name,
                cell_object.name,
            );
        }

        // Get the controller from the library if kind is provided
        if let Some(controller_kind) = cell_object.controller.as_ref().unwrap().kind.as_ref() {
            if let Ok(lib_controller) = get_from_library(controller_kind) {
                if let Ok(controller_from_lib) = Controller::from_json(&lib_controller.to_string())
                {
                    // Check size
                    cell_object.controller.as_mut().unwrap().size = controller_from_lib.size;
                    // append parameters from controller_from_lib to controller_parameters
                    overwritten_params.extend(merge_parameters(
                        &mut cell_object.controller.as_mut().unwrap().parameters,
                        &controller_from_lib.parameters,
                    )?);
                    if !overwritten_params.is_empty() {
                        let mut warning = format!(
                            "Some parameters from controller {} in cell {} were overwritten:",
                            cell_object.controller.as_ref().unwrap().name,
                            cell_object.name,
                        );
                        for param in overwritten_params.iter() {
                            warning.push_str(&format!(
                                "\n - {}(old value: {}, new value: {})",
                                param.0, param.1, param.2
                            ));
                        }
                        warn!("{}", warning);
                    }
                    // append required parameters from controller_from_lib to controller
                    if !controller_from_lib.required_parameters.is_empty() {
                        cell_object
                            .controller
                            .as_mut()
                            .unwrap()
                            .required_parameters
                            .extend(controller_from_lib.required_parameters);
                    }
                }
            }
        }

        // Check controller validity
        if cell_object.controller.as_ref().unwrap().size.is_none() {
            panic!(
                "Size not found for controller {} in cell {}",
                cell_object.controller.as_ref().unwrap().name,
                cell_object.name,
            );
        }

        // Get the required parameters for the resource
        if cell_object
            .controller
            .as_ref()
            .unwrap()
            .required_parameters
            .is_empty()
        {
            cell_object.controller.as_mut().unwrap().required_parameters = cell_object
                .controller
                .as_ref()
                .unwrap()
                .parameters
                .keys()
                .cloned()
                .collect();
            warn!(
                    "No required parameters found for controller {} in cell {}, using all parameters as required",
                    cell_object.controller.as_ref().unwrap().name,
                    cell_object.name,
                );
        }

        // Add controller required parameters to the registry
        if cell_object
            .controller
            .as_ref()
            .unwrap()
            .parameters
            .is_empty()
            && cell_object
                .controller
                .as_ref()
                .unwrap()
                .required_parameters
                .is_empty()
        {
            warn!(
                "No parameters found for controller {} in cell {}",
                cell_object.controller.as_ref().unwrap().name,
                cell_object.name,
            );
        } else {
            let mut filtered_parameters = ParameterList::new();
            for required_param in &cell_object.controller.as_ref().unwrap().required_parameters {
                if let Some(param_value) = cell_object
                    .controller
                    .as_ref()
                    .unwrap()
                    .parameters
                    .get(required_param)
                {
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else if let Some(param_value) = cell_object.parameters.get(required_param) {
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else if let Some(param_value) = fabric_object.parameters.get(required_param) {
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else {
                    panic!(
                        "Required parameter {} not found for controller {} in cell {}",
                        required_param,
                        cell_object.controller.as_ref().unwrap().name,
                        cell_object.name,
                    );
                }
            }
            cell_object.controller.as_mut().unwrap().parameters = filtered_parameters.clone();
            match add_parameters(&filtered_parameters, &mut parameter_list) {
                Ok(_) => (),
                Err(e) => {
                    panic!(
                        "Error with controller parameters for controller {} in cell {}: ({})",
                        cell_object.controller.as_ref().unwrap().name,
                        cell_object.name,
                        e
                    );
                }
            }
        }

        // generate a hash based on the required parameters for the controller
        let controller_hash = generate_hash(
            vec![cell_object.controller.as_ref().unwrap().name.clone()],
            &parameter_list,
        );
        resource_hashes.push(controller_hash.clone());

        // Check if two controllers with same parameters have the same hash
        if !implemented_controllers.contains_key(&controller_hash) {
            implemented_controllers.insert(
                controller_hash.clone(),
                cell_object.controller.as_ref().unwrap().clone(),
            );
            let output_controller_json = serde_json::json!({
                "name": cell_object.controller.as_ref().unwrap().name,
                "fingerprint": controller_hash,
                "parameters": cell_object.controller.as_ref().unwrap().parameters,
            });
            output_controllers.push(output_controller_json);
        }
        //todo!("Generate the controller RTL here");

        // RESOURCES
        let mut current_slot = 0;
        for resource_object in cell_object.resources.as_mut().unwrap().iter_mut() {
            let mut overwritten_params = Vec::new();

            // Get the resource from the resource pool
            if let Some(resource_entry) = resource_pool
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["name"].as_str().unwrap() == resource_object.name)
            {
                if let Ok(resource_from_pool) = Resource::from_json(&resource_entry.to_string()) {
                    // Check slot
                    if resource_object.slot.is_none() && resource_from_pool.slot.is_some() {
                        resource_object.slot = resource_from_pool.slot;
                    }
                    // Check size
                    if resource_object.size.is_none() && resource_from_pool.size.is_some() {
                        resource_object.size = resource_from_pool.size;
                    }
                    // append parameters from resource_from_pool to resource_parameters
                    overwritten_params.extend(merge_parameters(
                        &mut resource_object.parameters,
                        &resource_from_pool.parameters,
                    )?);
                    // append required parameters from resource_from_pool to resource
                    if !resource_from_pool.required_parameters.is_empty() {
                        resource_object
                            .required_parameters
                            .extend(resource_from_pool.required_parameters);
                    }
                    // check if the resource has kind
                    if resource_from_pool.kind.is_some() && resource_object.kind.is_none() {
                        resource_object.kind = resource_from_pool.kind;
                    }
                }
            }

            // Check if the resource kind is provided
            if resource_object.kind.is_none() {
                panic!(
                    "Kind not found for resource {} in cell {}",
                    resource_object.name, cell_object.name,
                );
            }

            // Get the resource from the library
            if let Some(resource_kind) = resource_object.kind.as_ref() {
                if let Ok(lib_resource) = get_from_library(resource_kind) {
                    if let Ok(resource_from_lib) = Resource::from_json(&lib_resource.to_string()) {
                        // Check slot
                        if resource_object.slot.is_none() && resource_from_lib.slot.is_some() {
                            resource_object.slot = resource_from_lib.slot;
                        }
                        // Check size
                        if resource_object.size.is_none() && resource_from_lib.size.is_some() {
                            resource_object.size = resource_from_lib.size;
                        }
                        // append parameters from resource_from_lib to resource_parameters
                        overwritten_params.extend(merge_parameters(
                            &mut resource_object.parameters,
                            &resource_from_lib.parameters,
                        )?);
                        if !overwritten_params.is_empty() {
                            let mut warning = format!(
                            "Some parameters from resource {} (slot {}) in cell {} were overwritten:",
                            resource_object.name,
                            resource_object.slot.unwrap(),
                            cell_object.name,
                        );
                            for param in overwritten_params.iter() {
                                warning.push_str(&format!(
                                    "\n - {}(old value: {}, new value: {})",
                                    param.0, param.1, param.2
                                ));
                            }
                            warn!("{}", warning);
                        }
                        // append required parameters from resource_from_lib to resource
                        if !resource_from_lib.required_parameters.is_empty() {
                            resource_object
                                .required_parameters
                                .extend(resource_from_lib.required_parameters);
                        }
                    }
                }
            }

            // Check resource validity
            if resource_object.slot.is_none() {
                resource_object.slot = Some(current_slot);
                current_slot += resource_object.size.unwrap();
            }

            // Get the required parameters for the resource
            if resource_object.required_parameters.is_empty() {
                resource_object.required_parameters =
                    resource_object.parameters.keys().cloned().collect();
                warn!(
                    "No required parameters found for resource {} (slot {}) in cell {}, using all parameters as required",
                    resource_object.name,
                    resource_object.slot.unwrap(),
                    cell_object.name,
                );
            }

            // Add required parameters to the registry
            if resource_object.required_parameters.is_empty() {
                warn!(
                    "No parameters found for resource {} (slot {}) in cell {}",
                    resource_object.name,
                    resource_object.slot.unwrap(),
                    cell_object.name,
                );
            } else {
                // Filter parameters that are not required
                let mut filtered_parameters = ParameterList::new();
                for required_param in &resource_object.required_parameters {
                    if let Some(param_value) = resource_object.parameters.get(required_param) {
                        filtered_parameters.insert(required_param.clone(), *param_value);
                    } else if let Some(param_value) = cell_object.parameters.get(required_param) {
                        filtered_parameters.insert(required_param.clone(), *param_value);
                    } else if let Some(param_value) = fabric_object.parameters.get(required_param) {
                        filtered_parameters.insert(required_param.clone(), *param_value);
                    } else {
                        panic!(
                            "Required parameter {} not found for resource {} (slot {}) in cell {}",
                            required_param,
                            resource_object.name,
                            resource_object.slot.unwrap(),
                            cell_object.name,
                        );
                    }
                }
                resource_object.parameters = filtered_parameters.clone();
                match add_parameters(&filtered_parameters, &mut parameter_list) {
                    Ok(_) => (),
                    Err(e) => {
                        panic!(
                        "Error with resource parameters for resource ({}, slot {}) in cell {}: ({})",
                        resource_object.name,
                        resource_object.slot.unwrap(),
                        &cell_object.name,
                        e
                    );
                    }
                }
            }

            // generate a hash based on the required parameters for the resource
            let resource_hash = generate_hash(vec![resource_object.name.clone()], &parameter_list);
            resource_hashes.push(resource_hash.clone());

            // Check if two resources with same parameters have the same hash
            if !implemented_resources.contains_key(&resource_hash) {
                implemented_resources.insert(resource_hash.clone(), resource_object.clone());
                //todo!("Generate the resource RTL here");
                let output_resource_json = serde_json::json!({
                    "name": resource_object.name,
                    "fingerprint": resource_hash,
                    "parameters": resource_object.parameters,
                });
                output_resources.push(output_resource_json);
            }

            // Remove the resource parameters from the registry
            remove_parameters(&resource_object.parameters, &mut parameter_list).unwrap();
        }
        // generate a hash based on the required parameters for the cell
        let cell_hash = generate_hash(resource_hashes.clone(), &parameter_list);

        // Check if two cells with same parameters have the same hash
        if !implemented_cells.contains_key(&cell_hash) {
            implemented_cells.insert(cell_hash.clone(), cell_object.clone());
            // Create the json object for the cell
            let output_cell_json = serde_json::json!({
                "fingerprint": cell_hash,
                "name": cell_object.name,
                "parameters": cell_object.parameters,
                "resources": cell_object.get_resources_names(),
                "controller": cell_object.controller.as_ref().unwrap().name,

            });
            output_cells.push(output_cell_json);
        }
        //todo!("Generate the cell RTL here");

        // Add the cell to the fabric at the different coordinates
        for (row, col) in cell_object.coordinates_list.iter() {
            fabric_object.add_cell(&cell_object, *row, *col);
        }

        // Create the json object for the cell list
        for coordinate in cell_object.coordinates_list.iter() {
            let output_cell_json = serde_json::json!({
                "coordinates": coordinate,
                "cell": cell_object.name,
            });
            output_cells_list.push(output_cell_json);
        }

        // remove cell parameters from the registry
        remove_parameters(&cell_object.parameters, &mut parameter_list).unwrap();
    }

    let output_json = serde_json::json!({
        "cells": output_cells,
        "resources": output_resources,
        "controllers": output_controllers,
        "fabric": {
            "height": fabric_object.height,
            "width": fabric_object.width,
            "parameters": fabric_object.parameters,
            "cells_list": output_cells_list,
        },
    });

    // Write the output json to a file in the build directory
    let output_file = Path::new(&build_dir).join("fabric.json");
    fs::create_dir_all(build_dir).expect("Failed to create build directory");
    fs::write(
        &output_file,
        serde_json::to_string_pretty(&output_json).unwrap(),
    )?;
    println!("{}", serde_json::to_string_pretty(&output_json).unwrap());

    // 5. Generate the top module for fabric using the template and parameters

    Ok(())
}

fn get_parameters(component: &serde_json::Value, param_key: Option<String>) -> ParameterList {
    let param_key = param_key.unwrap_or("parameters".to_string());
    let mut parameters = ParameterList::new();
    let component_params = component.get(param_key);
    if let Some(component_params) = component_params {
        for param in component_params.as_array().unwrap() {
            let name = param.get("name").unwrap().as_str().unwrap();
            let value = param.get("value").unwrap();
            parameters.insert(name.to_string(), value.as_u64().unwrap());
        }
    }
    parameters
}

/// Merge parameters from params2 into params1
fn merge_parameters(
    params1: &mut ParameterList,
    params2: &ParameterList,
) -> Result<Vec<(String, u64, u64)>> {
    let mut overwritten_params = Vec::new();
    for (param_name, param_value) in params2.iter() {
        // Check if the parameter already exists in params1
        if !params1.contains_key(param_name) {
            params1.insert(param_name.clone(), *param_value);
        } else {
            let existing_param = params1.get(param_name).unwrap();
            if existing_param != param_value {
                // List the parameters that exist in param1 but with a different value
                overwritten_params.push((param_name.clone(), *existing_param, *param_value));
            }
        }
    }
    Ok(overwritten_params)
}

fn get_from_library(component_name: &String) -> Result<serde_json::Value> {
    // Get the library path
    let library_path = get_library_path();

    // Check if a folder in the library is named the same as the cell
    let mut cell_path = None;
    for entry in walkdir::WalkDir::new(&library_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        debug!("Checking path: {:?}", entry.path());
        let path = entry.path();
        if path.is_dir()
            && path
                .file_name()
                .map_or(false, |name| name == component_name.as_str())
        {
            cell_path = Some(entry.path().to_path_buf());
            break;
        }
    }

    if cell_path.is_none() {
        error!("Component {} not found in the library", component_name);
        Err(Error::new(
            std::io::ErrorKind::NotFound,
            format!("Component {} not found in the library", component_name),
        ))
    } else {
        let cell_path = cell_path.unwrap();
        // Get the arch.json file for the cell
        let arch_path = cell_path.join("arch.json");
        if !arch_path.exists() {
            warn!(
                "Component \"{}\" JSON description not found in library (component path: {})",
                component_name,
                cell_path.to_str().unwrap()
            );
            return Err(Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Component {} does not contain an arch.json file",
                    component_name
                ),
            ));
        }

        // Read the arch.json file
        let json_str = std::fs::read_to_string(&arch_path).expect("Failed to read file");
        let component_result = serde_json::from_str(&json_str);
        match component_result {
            Ok(component) => Ok(component),
            Err(_) => {
                warn!(
                    "Failed to parse JSON description for component \"{}\" (component path: {})",
                    component_name,
                    arch_path.to_str().unwrap()
                );
                Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to parse json file: {}", arch_path.to_str().unwrap()),
                ))
            }
        }
    }
}

fn get_library_path() -> String {
    let lib_path = env::var("VESYLA_LIBRARY_PATH").expect("Environment variable VESYLA_LIBRARY_PATH not set! Did you forget to source the setup script env.sh?");
    // get abosulte path
    let abosulte = std::path::absolute(lib_path).expect("Cannot get absolute path for library");
    abosulte.to_str().unwrap().to_string()
}

fn generate_hash(names: Vec<String>, parameters: &BTreeMap<String, u64>) -> String {
    let mut hasher = DefaultHasher::new();
    for name in names {
        hasher.write(name.as_bytes());
    }
    for (param_name, param_value) in parameters.iter() {
        hasher.write(param_name.as_bytes());
        hasher.write(&param_value.to_be_bytes());
    }
    let hash = hasher.finish();
    encode(hash.to_be_bytes()).into_string()
}

fn add_parameters(parameters: &ParameterList, parameter_list: &mut ParameterList) -> Result<()> {
    for (param_name, param_value) in parameters.iter() {
        if add_parameter(param_name.to_string(), *param_value, parameter_list).is_err() {
            return Err(Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("Duplicate parameter: {}", param_name),
            ));
        }
    }
    Ok(())
}

fn add_parameter(key: String, value: u64, parameter_list: &mut ParameterList) -> Result<()> {
    if parameter_list.contains_key(key.as_str())
        && parameter_list.get(key.as_str()).unwrap() != &value
    {
        return Err(Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "Duplicate parameter with different value: {} ({} vs. {})",
                key,
                parameter_list.get(key.as_str()).unwrap(),
                value
            ),
        ));
    }

    parameter_list.insert(key, value);

    Ok(())
}

fn remove_parameters(parameters: &ParameterList, parameter_list: &mut ParameterList) -> Result<()> {
    for (param_name, _) in parameters.iter() {
        parameter_list.remove(param_name);
    }
    Ok(())
}

fn gen_api(input_file: String, output_file: String) {
    // parse the "args" using argparse
    // -i <input file> -o <output file>
    //let mut input_file = String::from("isa.json");
    //let mut output_file = String::from("api.sv");
    //{
    //    let mut ap = argparse::ArgumentParser::new();
    //    ap.set_description("Generate system verilog api");
    //    ap.refer(&mut input_file)
    //        .add_option(&["-i", "--input"], argparse::Store, "Input file");
    //    ap.refer(&mut output_file)
    //        .add_option(&["-o", "--output"], argparse::Store, "Output file");
    //    ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr())
    //        .unwrap();
    //}

    generator::gen_api(input_file, output_file);
}

fn gen_doc(input_file: String, output_file: String) {
    // parse the "args" using argparse
    // -i <input file> -o <output file>
    //let mut input_file: String = String::from("isa.json");
    //let mut output_file: String = String::from("doc.md");
    //{
    //    let mut ap = argparse::ArgumentParser::new();
    //    ap.set_description("Generate system verilog api");
    //    ap.refer(&mut input_file)
    //        .add_option(&["-i", "--input"], argparse::Store, "Input file");
    //    ap.refer(&mut output_file)
    //        .add_option(&["-o", "--output"], argparse::Store, "Output file");
    //    ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr())
    //        .unwrap();
    //}

    generator::gen_doc(input_file, output_file);
}

fn assemble(input_file: String, output_dir: String) {
    // parse the "args" using argparse
    // -i <input file> -o <output directory>
    //let mut input_file = String::from("conf.json");
    //let mut output_dir = String::from(".");
    //{
    //    let mut ap = argparse::ArgumentParser::new();
    //    ap.set_description("Assemble the system");
    //    ap.refer(&mut input_file)
    //        .add_option(&["-i", "--input"], argparse::Store, "Input file");
    //    ap.refer(&mut output_dir).add_option(
    //        &["-o", "--output"],
    //        argparse::Store,
    //        "Output directory",
    //    );
    //    ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr())
    //        .unwrap();
    //}

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
        } else if path.file_name().unwrap() == "arch.json" {
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
        !existing_search_path_vec.is_empty(),
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
            panic!("Component {} not found in the component library", cell_name);
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
                panic!(
                    "Controller {} not found in the component library",
                    controller_name
                );
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
                        panic!(
                            "Resource {} not found in the component library",
                            resource_name
                        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_rtl() {
        let args = [env!("CARGO_MANIFEST_DIR").to_string() + "/simple_example.json"];
        std::env::set_var(
            "VESYLA_LIBRARY_PATH",
            env!("CARGO_MANIFEST_DIR").to_string() + "/template",
        );
        gen_rtl(args[0].clone(), "build".to_string()).unwrap();
    }
}
