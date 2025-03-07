use crate::drra;
use crate::drra::{Cell, Controller, Fabric, ParameterList, RTLComponent, Resource};
use crate::isa::InstructionSet;
use crate::utils::*;
use bs58::encode;
use log::{debug, info, warn};
use std::collections::HashMap;
use std::fs;
use std::hash::{DefaultHasher, Hasher};
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

fn generate_hash(names: Vec<String>, parameters: &ParameterList) -> String {
    let mut hasher = DefaultHasher::new();
    for name in names {
        hasher.write(name.as_bytes());
    }
    for (param_name, param_value) in parameters.iter() {
        hasher.write(param_name.as_bytes());
        hasher.write(&param_value.to_be_bytes());
    }
    let hash = hasher.finish();

    let str_hash = encode(hash.to_be_bytes()).into_string().to_lowercase();
    "_".to_string() + &str_hash
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
        let mut overwritten_params = Vec::new();
        // Get the controller from the controller pool
        if cell_object.controller.as_mut().unwrap().size.is_none() {
            if let Some(controller) = controller_pool.as_array().unwrap().iter().find(|entry| {
                entry["name"].as_str().unwrap() == cell_object.controller.as_ref().unwrap().name
            }) {
                if let Ok(controller_from_pool) = Controller::from_json(&controller.to_string()) {
                    // Check size
                    if cell_object.controller.as_ref().unwrap().size.is_none()
                        && controller_from_pool.size.is_some()
                    {
                        cell_object.controller.as_mut().unwrap().size = controller_from_pool.size;
                    }
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
                    // check io input
                    if controller_from_pool.io_input.is_some()
                        && cell_object.controller.as_ref().unwrap().io_input.is_none()
                    {
                        cell_object.controller.as_mut().unwrap().io_input =
                            controller_from_pool.io_input;
                    }
                    // check io output
                    if controller_from_pool.io_output.is_some()
                        && cell_object.controller.as_ref().unwrap().io_output.is_none()
                    {
                        cell_object.controller.as_mut().unwrap().io_output =
                            controller_from_pool.io_output;
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
        if let Some(controller_kind) = cell_object.controller.as_ref().unwrap().kind.clone() {
            if let Ok(lib_controller) = get_arch_from_library(&controller_kind.clone()) {
                if let Ok(controller_from_lib) = Controller::from_json(&lib_controller.to_string())
                {
                    // Check size
                    if cell_object.controller.as_ref().unwrap().size.is_none()
                        && controller_from_lib.size.is_some()
                    {
                        cell_object.controller.as_mut().unwrap().size = controller_from_lib.size;
                    }
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

                    // append parameters from controller_from_lib to required_parameters
                    for param_key in controller_from_lib.parameters.keys() {
                        if !cell_object
                            .controller
                            .as_ref()
                            .unwrap()
                            .required_parameters
                            .contains(param_key)
                        {
                            cell_object
                                .controller
                                .as_mut()
                                .unwrap()
                                .required_parameters
                                .push(param_key.clone());
                        }
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
                    // check io input
                    if controller_from_lib.io_input.is_some()
                        && cell_object.controller.as_ref().unwrap().io_input.is_none()
                    {
                        cell_object.controller.as_mut().unwrap().io_input =
                            controller_from_lib.io_input;
                    }
                    // check io output
                    if controller_from_lib.io_output.is_some()
                        && cell_object.controller.as_ref().unwrap().io_output.is_none()
                    {
                        cell_object.controller.as_mut().unwrap().io_output =
                            controller_from_lib.io_output;
                    }
                    // get ISA from library if is None
                    if cell_object.controller.as_ref().unwrap().isa.is_none() {
                        let isa_json = get_isa_from_library(&controller_kind.clone()).unwrap();
                        let isa = InstructionSet::from_json(isa_json);
                        if isa.is_err() {
                            panic!(
                                "Error with ISA for controller {} in cell {} from library (kind: {}) -> {}",
                                cell_object.controller.as_ref().unwrap().name,
                                cell_object.name,
                                controller_kind,
                                isa.err().unwrap()
                            );
                        }
                        cell_object.controller.as_mut().unwrap().isa = Some(isa.unwrap());
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
        if cell_object.controller.as_ref().unwrap().isa.is_none() {
            panic!(
                "ISA not found for controller {} in cell {} (kind: {})",
                cell_object.controller.as_ref().unwrap().name,
                cell_object.name,
                cell_object
                    .controller
                    .as_ref()
                    .unwrap()
                    .kind
                    .as_ref()
                    .unwrap(),
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
        let mut controller_added_parameters = ParameterList::new();
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
                Ok(added_params) => {
                    controller_added_parameters = added_params;
                }
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
            vec![cell_object
                .controller
                .as_ref()
                .unwrap()
                .kind
                .clone()
                .unwrap()],
            &parameter_list,
        );
        resource_hashes.push(controller_hash.clone());
        cell_object.controller.as_mut().unwrap().fingerprint = Some(controller_hash.clone());

        // Check if two controllers with same parameters have the same hash
        if !implemented_controllers.contains_key(&controller_hash) {
            implemented_controllers.insert(
                controller_hash.clone(),
                cell_object.controller.as_ref().unwrap().clone(),
            );
        } else {
            cell_object.controller.as_mut().unwrap().already_defined = true;
        }

        debug!(
            "Controller: {} in cell: {} (hash: {})",
            cell_object.controller.as_ref().unwrap().name,
            cell_object.name,
            controller_hash,
        );
        debug!(
            "Serialized controller: \n{}",
            serde_json::to_string_pretty(&cell_object.controller.as_ref()).unwrap()
        );

        // Check if the RTL has already been generated for this controller
        let rtl_output_file = Path::new(&rtl_output_dir).join(format!(
            "{}.sv",
            cell_object.controller.as_ref().unwrap().name
        ));
        if Path::exists(&rtl_output_file) {
            debug!(
                "Using existing RTL for controller {} (hash: {})",
                cell_object.controller.as_ref().unwrap().name,
                controller_hash,
            );
        } else {
            // Generate the controller RTL here
            if cell_object
                .controller
                .as_ref()
                .unwrap()
                .generate_rtl(&rtl_output_file)
                .is_ok()
            {
                debug!(
                    "Generated RTL for controller {} (hash: {})",
                    cell_object.controller.as_ref().unwrap().name,
                    controller_hash,
                );
            } else {
                panic!(
                    "Failed to generate RTL for controller {} (hash: {})",
                    cell_object.controller.as_ref().unwrap().name,
                    controller_hash,
                );
            }
        }

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
            let resource_hash =
                generate_hash(vec![resource_object.kind.clone().unwrap()], &parameter_list);
            resource_hashes.push(resource_hash.clone());
            resource_object.fingerprint = Some(resource_hash.clone());

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
            let rtl_output_file =
                Path::new(&rtl_output_dir).join(format!("{}.sv", resource_object.name));
            if Path::exists(&rtl_output_file) {
                debug!(
                    "Using existing RTL for resource {} (hash: {})",
                    resource_object.name, resource_hash,
                );
            } else {
                // Generate the resource RTL here
                if resource_object.generate_rtl(&rtl_output_file).is_ok() {
                    debug!(
                        "Generated RTL for resource {} (hash: {})",
                        resource_object.name, resource_hash,
                    );
                } else {
                    panic!(
                        "Failed to generate RTL for resource {} (hash: {})",
                        resource_object.name, resource_hash,
                    );
                }
            }

            // Remove the resource parameters from the registry
            remove_parameters(&resource_added_parameters, &mut parameter_list).unwrap();
        }
        // generate a hash based on the required parameters for the cell
        let mut cell_hash_content = Vec::new();
        if cell_object.kind.is_none() {
            cell_hash_content = resource_hashes.clone();
        } else {
            cell_hash_content.push(cell_object.kind.clone().unwrap());
            cell_hash_content.extend(resource_hashes.clone());
        }
        let cell_hash = generate_hash(cell_hash_content, &parameter_list);
        cell_object.fingerprint = Some(cell_hash.clone());

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

        let rtl_output_file = Path::new(&rtl_output_dir).join(format!("{}.sv", cell_object.name));
        // Check if the RTL has already been generated for this cell
        if Path::exists(&rtl_output_file) {
            debug!(
                "Using existing RTL for cell {} (hash: {})",
                cell_object.name, cell_hash,
            );
        } else {
            debug!(
                "Serialized cell: \n{}",
                serde_json::to_string_pretty(&cell_object).unwrap()
            );
            if cell_object.generate_rtl(&rtl_output_file).is_ok() {
                debug!(
                    "Generated RTL for cell {} (hash: {})",
                    cell_object.name, cell_hash,
                );
            } else {
                panic!(
                    "Failed to generate RTL for cell {} (hash: {})",
                    cell_object.name, cell_hash,
                );
            }
        }
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
    let fabric_output_file = Path::new(&rtl_output_dir).join("fabric.sv");
    if fabric_object.generate_rtl(&fabric_output_file).is_ok() {
        debug!("Generated RTL for fabric");
    } else {
        panic!("Failed to generate RTL for fabric");
    }

    // copy .sv files in utils directory to the output directory
    let utils_string = String::from("utils");
    let utils_dir = get_path_from_library(&utils_string).unwrap();
    for entry in fs::read_dir(utils_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if file_name.ends_with(".sv") {
                let output_file = Path::new(&rtl_output_dir).join(file_name);
                fs::copy(&path, &output_file).unwrap();
            }
        }
    }

    Ok(())
}
