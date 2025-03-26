use crate::drra;
use crate::drra::{Cell, Controller, Fabric, ParameterList, RTLComponent, Resource};
use crate::isa::InstructionSet;
use crate::utils::*;
use log::{debug, info, warn};
use std::collections::HashMap;
use std::fs;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};

pub fn get_rtl_output_dir(build_dir: &String) -> Result<PathBuf> {
    let rtl_output_dir = Path::new(&build_dir).join("rtl");
    fs::create_dir_all(&rtl_output_dir)?;
    Ok(rtl_output_dir)
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

fn add_parameters(
    parameters_to_add: &ParameterList,
    parameter_list_destination: &mut ParameterList,
) -> Result<ParameterList> {
    let mut added_params = ParameterList::new();
    for (param_name, param_value) in parameters_to_add.iter() {
        match add_parameter(
            param_name.to_string(),
            *param_value,
            parameter_list_destination,
        ) {
            Ok(true) => {
                added_params.insert(param_name.to_string(), *param_value);
            }
            Ok(false) => (),
            Err(_) => {
                return Err(Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("Duplicate parameter: {}", param_name),
                ));
            }
        }
    }
    Ok(added_params)
}

fn add_parameter(key: String, value: u64, parameter_list: &mut ParameterList) -> Result<bool> {
    if parameter_list.contains_key(key.as_str()) {
        if parameter_list.get(key.as_str()).unwrap() != &value {
            return Err(Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "Duplicate parameter with different value: {} ({} vs. {})",
                    key,
                    parameter_list.get(key.as_str()).unwrap(),
                    value
                ),
            ));
        } else {
            return Ok(false);
        }
    }

    parameter_list.insert(key, value);
    Ok(true)
}

fn remove_parameters(parameters: &ParameterList, parameter_list: &mut ParameterList) -> Result<()> {
    for (param_name, _) in parameters.iter() {
        parameter_list.remove(param_name);
    }
    Ok(())
}

pub fn gen_rtl(fabric_filepath: &String, build_dir: &String, output_json: &String) -> Result<()> {
    // Clean build directory
    let rtl_output_dir = get_rtl_output_dir(build_dir)?;

    // Create lists for implemented cells, resources and controllers
    let mut implemented_cells: HashMap<String, Cell> = HashMap::new();
    let mut implemented_resources: HashMap<String, Resource> = HashMap::new();
    let mut implemented_controllers: HashMap<String, Controller> = HashMap::new();

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

    // Create the fabric object
    let mut fabric_object = Fabric::new();
    fabric_object.add_parameters(get_parameters(
        fabric,
        Some("custom_properties".to_string()),
    ));

    // add fabric parameters to the registry
    if fabric_object.parameters.is_empty() {
        warn!("No fabric parameters found in {}", fabric_filepath);
    } else {
        match add_parameters(&fabric_object.parameters, &mut parameter_list) {
            Ok(_) => (),
            Err(e) => {
                panic!("Error with fabric.json parameters: ({})", e);
            }
        }
    }

    // CELLS
    let cell_list = fabric.get("cells_list").expect("Cells not found in fabric");
    for cell in cell_list.as_array().unwrap() {
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
                // Check io input
                if cell_from_pool.io_input.is_some() && cell_object.io_input.is_none() {
                    cell_object.io_input = cell_from_pool.io_input;
                }
                // Check io output
                if cell_from_pool.io_output.is_some() && cell_object.io_output.is_none() {
                    cell_object.io_output = cell_from_pool.io_output;
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
            if let Ok(lib_cell) = get_arch_from_library(cell_type) {
                if let Ok(cell_from_lib) = Cell::from_json(&lib_cell.to_string()) {
                    // If cell controller was not provided in "fabric" or cell pool, get from cell_from_lib
                    if cell_from_lib.controller.is_some() && cell_object.controller.is_none() {
                        cell_object.controller = cell_from_lib.controller.clone();
                    }
                    // If cell resources were not provided in "fabric" or cell pool, get from cell_from_lib
                    if cell_from_lib.resources.is_some() && cell_object.resources.is_none() {
                        cell_object.resources = cell_from_lib.resources.clone();
                    }
                    // Get ISA from library if is None
                    if cell_object.isa.is_none() {
                        let isa_json = get_isa_from_library(cell_type).unwrap();
                        let isa = InstructionSet::from_json(isa_json);
                        if isa.is_err() {
                            panic!(
                                "Error with ISA for cell {} from library (kind: {}) -> {}",
                                cell_object.name,
                                cell_type,
                                isa.err().unwrap()
                            );
                        }
                        cell_object.isa = Some(isa.unwrap());
                    }
                    // Get parameters for cell_object

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

                    // append parameters from cell_from_lib to required_parameters
                    for param_key in cell_from_lib.parameters.keys() {
                        if !cell_object.required_parameters.contains(param_key) {
                            cell_object.required_parameters.push(param_key.clone());
                        }
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
        if cell_object.isa.is_none() {
            panic!(
                "ISA not found for cell {} (kind: {})",
                cell_object.name,
                cell_object.kind.as_ref().unwrap(),
            );
        }
        if cell_object.resources.is_none() {
            panic!("Resources for cell {} not found in the library and were not provided in JSON file fabric or cell pool", cell_object.name);
        }
        let mut cell_added_parameters = ParameterList::new();
        if cell_object.parameters.is_empty() && cell_object.required_parameters.is_empty() {
            warn!(
                "No parameters or required_parameters found for cell {}",
                cell_object.name,
            );
        } else {
            // Find and add the required parameters to the cell
            let mut filtered_parameters = ParameterList::new();
            for required_param in &cell_object.required_parameters {
                if let Some(param_value) = cell_object.parameters.get(required_param) {
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else if let Some(param_value) = fabric_object.parameters.get(required_param) {
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else {
                    panic!(
                        "Required parameter {} not found for cell {}",
                        required_param, cell_object.name,
                    );
                }
            }
            cell_object.parameters = filtered_parameters.clone();
            // Add cell parameters to the registry
            match add_parameters(&cell_object.parameters, &mut parameter_list) {
                Ok(added_params) => {
                    cell_added_parameters = added_params;
                }
                Err(e) => {
                    panic!(
                        "Error with cell parameters for cell ({}): {}",
                        cell_object.name, e,
                    );
                }
            }
        }

        // CONTROLLER
        let controller_object = cell_object.controller.as_mut().unwrap();
        let mut overwritten_params = Vec::new();
        // Get the controller from the controller pool
        if controller_object.size.is_none() {
            if let Some(controller) = controller_pool
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["name"].as_str().unwrap() == controller_object.name)
            {
                if let Ok(controller_from_pool) = Controller::from_json(&controller.to_string()) {
                    // Check size
                    if controller_object.size.is_none() && controller_from_pool.size.is_some() {
                        controller_object.size = controller_from_pool.size;
                    }
                    // append parameters from controller_from_pool to controller_parameters
                    overwritten_params.extend(merge_parameters(
                        &mut controller_object.parameters,
                        &controller_from_pool.parameters,
                    )?);
                    // append required parameters from controller_from_pool to controller
                    if !controller_from_pool.required_parameters.is_empty() {
                        controller_object
                            .required_parameters
                            .extend(controller_from_pool.required_parameters);
                    }
                    // check if the controller has kind
                    if controller_from_pool.kind.is_some() && controller_object.kind.is_none() {
                        controller_object.kind = controller_from_pool.kind;
                    }
                    // check io input
                    if controller_from_pool.io_input.is_some()
                        && controller_object.io_input.is_none()
                    {
                        controller_object.io_input = controller_from_pool.io_input;
                    }
                    // check io output
                    if controller_from_pool.io_output.is_some()
                        && controller_object.io_output.is_none()
                    {
                        controller_object.io_output = controller_from_pool.io_output;
                    }
                }
            }
        }

        // Check if the controller has kind
        if controller_object.kind.is_none() {
            panic!(
                "Kind not found for controller {} in cell {}",
                controller_object.name, cell_object.name,
            );
        }

        // Get the controller from the library if kind is provided
        if let Some(controller_kind) = controller_object.kind.clone() {
            if let Ok(lib_controller) = get_arch_from_library(&controller_kind.clone()) {
                if let Ok(controller_from_lib) = Controller::from_json(&lib_controller.to_string())
                {
                    // Check size
                    if controller_object.size.is_none() && controller_from_lib.size.is_some() {
                        controller_object.size = controller_from_lib.size;
                    }
                    // append parameters from controller_from_lib to controller_parameters
                    overwritten_params.extend(merge_parameters(
                        &mut controller_object.parameters,
                        &controller_from_lib.parameters,
                    )?);
                    if !overwritten_params.is_empty() {
                        let mut warning = format!(
                            "Some parameters from controller {} in cell {} were overwritten:",
                            controller_object.name, cell_object.name,
                        );
                        for param in overwritten_params.iter() {
                            warning.push_str(&format!(
                                "\n - {}(old value: {}, new value: {})",
                                param.0, param.1, param.2
                            ));
                        }
                        warn!("{}", warning);
                    }

                    // append parameters from controller_from_lib to required_parameters
                    for param_key in controller_from_lib.parameters.keys() {
                        if !controller_object.required_parameters.contains(param_key) {
                            controller_object
                                .required_parameters
                                .push(param_key.clone());
                        }
                    }

                    // append required parameters from controller_from_lib to controller
                    if !controller_from_lib.required_parameters.is_empty() {
                        controller_object
                            .required_parameters
                            .extend(controller_from_lib.required_parameters);
                    }
                    // check io input
                    if controller_from_lib.io_input.is_some()
                        && controller_object.io_input.is_none()
                    {
                        controller_object.io_input = controller_from_lib.io_input;
                    }
                    // check io output
                    if controller_from_lib.io_output.is_some()
                        && controller_object.io_output.is_none()
                    {
                        controller_object.io_output = controller_from_lib.io_output;
                    }
                    // get ISA from library if is None
                    if controller_object.isa.is_none() {
                        let isa_json = get_isa_from_library(&controller_kind.clone()).unwrap();
                        let isa = InstructionSet::from_json(isa_json);
                        if isa.is_err() {
                            panic!(
                                "Error with ISA for controller {} in cell {} from library (kind: {}) -> {}",
                                controller_object.name,
                                cell_object.name,
                                controller_kind,
                                isa.err().unwrap()
                            );
                        }
                        controller_object.isa = Some(isa.unwrap());
                    }
                }
            }
        }

        // Check controller validity
        if controller_object.size.is_none() {
            panic!(
                "Size not found for controller {} in cell {}",
                controller_object.name, cell_object.name,
            );
        }
        if controller_object.isa.is_none() {
            panic!(
                "ISA not found for controller {} in cell {} (kind: {})",
                controller_object.name,
                cell_object.name,
                controller_object.kind.as_ref().unwrap(),
            );
        }

        // Get the required parameters for the resource
        if controller_object.required_parameters.is_empty() {
            controller_object.required_parameters =
                controller_object.parameters.keys().cloned().collect();
            warn!(
                    "No required parameters found for controller {} in cell {}, using all parameters as required",
                    controller_object.name,
                    cell_object.name,
                );
        }

        // Add controller required parameters to the registry
        let mut controller_added_parameters = ParameterList::new();
        if controller_object.parameters.is_empty()
            && controller_object.required_parameters.is_empty()
        {
            warn!(
                "No parameters found for controller {} in cell {}",
                controller_object.name, cell_object.name,
            );
        } else {
            let mut filtered_parameters = ParameterList::new();
            for required_param in &controller_object.required_parameters {
                if let Some(param_value) = controller_object.parameters.get(required_param) {
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else if let Some(param_value) = cell_object.parameters.get(required_param) {
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else if let Some(param_value) = fabric_object.parameters.get(required_param) {
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else {
                    panic!(
                        "Required parameter {} not found for controller {} in cell {}",
                        required_param, controller_object.name, cell_object.name,
                    );
                }
            }
            controller_object.parameters = filtered_parameters.clone();
            match add_parameters(&filtered_parameters, &mut parameter_list) {
                Ok(added_params) => {
                    controller_added_parameters = added_params;
                }
                Err(e) => {
                    panic!(
                        "Error with controller parameters for controller {} in cell {}: ({})",
                        controller_object.name, cell_object.name, e
                    );
                }
            }
        }

        // generate a hash based on the required parameters for the controller
        let controller_hash = controller_object.get_fingerprint();

        // Check if two controllers with same parameters have the same hash
        if !implemented_controllers.contains_key(&controller_hash.clone()) {
            implemented_controllers
                .insert(controller_hash.clone(), controller_object.clone().clone());
        } else {
            controller_object.already_defined = true;
        }
        debug!(
            "Controller: {} in cell: {} (hash: {})",
            controller_object.name, cell_object.name, controller_hash,
        );
        debug!(
            "Serialized controller: \n{}",
            serde_json::to_string_pretty(&controller_object).unwrap()
        );

        // Generate RTL
        let controller_with_hash = format!("{}_{}", controller_object.name, controller_hash);
        let bender_output_folder = Path::new(&rtl_output_dir)
            .join("controllers")
            .join(controller_with_hash);
        let rtl_output_folder = bender_output_folder.join("rtl");

        // Generate the controller RTL here
        if !controller_object.already_defined {
            controller_object
                .generate_rtl(&rtl_output_folder)
                .expect("Failed to generate RTL for controller");
        }
        controller_object
            .generate_bender(&bender_output_folder)
            .unwrap();

        // Remove the controller parameters from the registry
        remove_parameters(&controller_added_parameters, &mut parameter_list).unwrap();

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
                    // check io input
                    if resource_from_pool.io_input.is_some() && resource_object.io_input.is_none() {
                        resource_object.io_input = resource_from_pool.io_input;
                    }
                    // check io output
                    if resource_from_pool.io_output.is_some() && resource_object.io_output.is_none()
                    {
                        resource_object.io_output = resource_from_pool.io_output;
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
                if let Ok(lib_resource) = get_arch_from_library(resource_kind) {
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
                        // if resource_object.parameters.is_empty() {
                        //     warn!(
                        //         "No parameters found for resource {} (slot {}) in cell {}",
                        //         resource_object.name,
                        //         resource_object.slot.unwrap(),
                        //         cell_object.name,
                        //     );
                        // } else {
                        //     debug!(
                        //         "Parameters for resource {} (slot {}) in cell {}: {:?}",
                        //         resource_object.name,
                        //         resource_object.slot.unwrap(),
                        //         cell_object.name,
                        //         resource_object.parameters
                        //     );
                        // }
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

                        // append parameters from resource_from_lib to required_parameters
                        for param_key in resource_from_lib.parameters.keys() {
                            if !resource_object.required_parameters.contains(param_key) {
                                resource_object.required_parameters.push(param_key.clone());
                            }
                        }

                        // append required parameters from resource_from_lib to resource
                        if !resource_from_lib.required_parameters.is_empty() {
                            resource_object
                                .required_parameters
                                .extend(resource_from_lib.required_parameters);
                        }
                        // check io input
                        if resource_from_lib.io_input.is_some()
                            && resource_object.io_input.is_none()
                        {
                            resource_object.io_input = resource_from_lib.io_input;
                        }
                        // check io output
                        if resource_from_lib.io_output.is_some()
                            && resource_object.io_output.is_none()
                        {
                            resource_object.io_output = resource_from_lib.io_output;
                        }
                        // get ISA from library if is None
                        if resource_object.isa.is_none() {
                            let isa_json = get_isa_from_library(resource_kind).unwrap();
                            let isa = InstructionSet::from_json(isa_json);
                            if isa.is_err() {
                                panic!(
                                    "Error with ISA for resource {} (slot {}) in cell {} from library (kind: {}) -> {}",
                                    resource_object.name,
                                    resource_object.slot.unwrap(),
                                    cell_object.name,
                                    resource_kind,
                                    isa.err().unwrap()
                                );
                            }
                            resource_object.isa = Some(isa.unwrap());
                        }
                    }
                }
            }

            // Check resource validity
            if resource_object.slot.is_none() {
                resource_object.slot = Some(current_slot);
                current_slot += resource_object.size.unwrap();
            }
            if resource_object.isa.is_none() {
                panic!(
                    "ISA not found for resource {} (slot {}) in cell {} (kind: {})",
                    resource_object.name,
                    resource_object.slot.unwrap(),
                    cell_object.name,
                    resource_object.kind.as_ref().unwrap(),
                );
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
            let mut resource_added_parameters = ParameterList::new();
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
                    Ok(added_params) => {
                        resource_added_parameters = added_params;
                    }
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
            let resource_hash = resource_object.get_fingerprint();

            // Check if two resources with same parameters have the same hash
            if !implemented_resources.contains_key(&resource_hash) {
                implemented_resources.insert(resource_hash.clone(), resource_object.clone());
            } else {
                resource_object.already_defined = true;
            }

            debug!(
                "Resource: {} (slot: {}) in cell: {} (hash: {})",
                resource_object.name,
                resource_object.slot.unwrap(),
                cell_object.name,
                resource_hash,
            );
            debug!(
                "Serialized resource: \n{}",
                serde_json::to_string_pretty(&resource_object).unwrap()
            );

            // Check if the RTL has already been generated for this resource
            let resource_with_hash = format!("{}_{}", resource_object.name, resource_hash);
            let bender_output_folder = Path::new(&rtl_output_dir)
                .join("resources")
                .join(resource_with_hash);
            let rtl_output_folder = bender_output_folder.join("rtl");

            // Generate the resource RTL
            if !resource_object.already_defined {
                resource_object
                    .generate_rtl(&rtl_output_folder)
                    .expect("Failed to generate RTL for resource");
            }
            resource_object
                .generate_bender(&bender_output_folder)
                .unwrap();

            // Remove the resource parameters from the registry
            remove_parameters(&resource_added_parameters, &mut parameter_list).unwrap();
        }

        let cell_hash = cell_object.get_fingerprint();

        // Check if two cells with same parameters have the same hash
        if !implemented_cells.contains_key(&cell_hash) {
            implemented_cells.insert(cell_hash.clone(), cell_object.clone());
        } else {
            cell_object.already_defined = true;
        }

        // Add the cell to the fabric at the different coordinates
        for (row, col) in cell_object.coordinates_list.iter() {
            fabric_object.add_cell(&cell_object, *row, *col);
        }

        let cell_with_hash = format!("{}_{}", cell_object.name, cell_hash);
        let bender_output_folder = Path::new(&rtl_output_dir)
            .join("cells")
            .join(cell_with_hash);
        let rtl_output_folder = bender_output_folder.join("rtl");

        // Generate the cell RTL
        if !cell_object.already_defined {
            cell_object
                .generate_rtl(&rtl_output_folder)
                .expect("Failed to generate RTL for cell");
        }
        cell_object.generate_bender(&bender_output_folder).unwrap();

        // remove cell parameters from the registry
        remove_parameters(&cell_added_parameters, &mut parameter_list).unwrap();
    }

    debug!(
        "Serialized fabric: \n{}",
        serde_json::to_string_pretty(&fabric_object).unwrap()
    );
    // Output the fabric object to a JSON file
    if !output_json.is_empty() {
        // create output directory if it does not exist
        let output_dir = Path::new(output_json).parent().unwrap();
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
        let fabric_output_file = Path::new(&output_json);
        let fabric_output = fs::File::create(fabric_output_file).expect("Failed to create file");
        serde_json::to_writer_pretty(fabric_output, &fabric_object).expect("Failed to write JSON");
        info!(
            "Generated JSON file for fabric at path: {}",
            fabric_output_file.display()
        );
    }

    // Output the fabric RTL
    let bender_output_folder = Path::new(&rtl_output_dir).join("fabric");
    // Remove files in the output directory
    if bender_output_folder.exists() {
        fs::remove_dir_all(&bender_output_folder).expect("Failed to remove output directory");
    }
    let rtl_output_folder = bender_output_folder.join("rtl");
    fabric_object
        .generate_rtl(&rtl_output_folder)
        .expect("Failed to generate RTL for fabric");
    fabric_object
        .generate_bender(&bender_output_folder)
        .unwrap();

    // copy .sv files in utils directory to the output directory
    let common_dir = get_path_from_library(&"common".to_string()).unwrap();
    for entry in fs::read_dir(common_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            let rtl_path = path.join("rtl");
            debug!("Checking directory: {:?}", rtl_path);
            // if contains a Bender.yml
            let bender_yml = path.join("Bender.yml");
            if bender_yml.exists() {
                debug!("Found Bender.yml in directory: {:?}", path);
                // Copy the directory to the output directory/common
                let output_dir = Path::new(&rtl_output_dir).join("common");
                if !output_dir.exists() {
                    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
                }
                let output_dir = output_dir.join(path.file_name().unwrap());
                if !output_dir.exists() {
                    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
                }
                // Copy bender file
                let bender_yml_output = output_dir.join("Bender.yml");
                debug!("Copying file: {:?} to {:?}", bender_yml, bender_yml_output);
                fs::copy(&bender_yml, &bender_yml_output).expect("Failed to copy file");
                let output_dir = output_dir.join("rtl");
                debug!("Copying directory: {:?} to {:?}", rtl_path, output_dir);
                copy_dir(&rtl_path, &output_dir).expect("Failed to copy directory");
            }
        }
    }

    Ok(())
}
