use crate::models::{
    cell::Cell,
    controller::Controller,
    drra::Fabric,
    resource::Resource,
    types::{DRRAError, ParameterList, ParameterListExt, RTLComponent},
};
use crate::utils::{copy_rtl_dir, get_arch_from_library, get_parameters, get_path_from_library};

use log::{debug, info, warn};
use std::{
    collections::HashMap,
    fmt::Write,
    fs,
    io::Error,
    path::{Path, PathBuf},
};

pub struct Pools<'a> {
    pub cell_pool: &'a serde_json::Value,
    pub resource_pool: &'a serde_json::Value,
    pub controller_pool: &'a serde_json::Value,
}

pub struct ImplementedObjects<'a> {
    pub implemented_cells: &'a mut HashMap<String, Cell>,
    pub implemented_resources: &'a mut HashMap<String, Resource>,
    pub implemented_controllers: &'a mut HashMap<String, Controller>,
}

pub fn get_rtl_output_dir(build_dir: &String) -> Result<PathBuf, Error> {
    let rtl_output_dir = Path::new(&build_dir).join("rtl");
    fs::create_dir_all(&rtl_output_dir)?;
    Ok(rtl_output_dir)
}

pub fn process_controller(
    cell_parameters: &ParameterList,
    controller_object: &mut Controller,
    controller_pool: &serde_json::Value,
    fabric_object: &Fabric,
    rtl_output_dir: &Path,
    parameter_list: &mut ParameterList,
    implemented_controllers: &mut HashMap<String, Controller>,
) -> Result<(), Error> {
    let mut overwritten_params = Vec::new();
    if controller_object.size.is_none() {
        if let Some(controller) = controller_pool
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["name"].as_str().unwrap() == controller_object.name)
        {
            overwritten_params.extend(
                controller_object
                    .update_from_json(&controller.to_string(), false)
                    .expect("Failed to update controller from JSON entry"),
            );
        }
    }

    // Get the controller from the library if kind is provided
    let Some(controller_kind) = controller_object.kind.as_ref() else {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Controller kind not found for controller {:?}",
                controller_object.name,
            ),
        ));
    };

    let lib_controller_arch = get_arch_from_library(controller_kind, None).map_err(|e| {
        Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Controller {} not found in library (kind: {}): {}",
                controller_object.name, controller_kind, e
            ),
        )
    })?;

    // Merge parameters from library to controller
    overwritten_params.extend(
        controller_object
            .update_from_json(&lib_controller_arch.to_string(), false)
            .expect("Failed to update controller from JSON entry"),
    );
    if !overwritten_params.is_empty() {
        let warning = overwritten_params.iter().fold(
            format!(
                "Some parameters from controller {} were overwritten:",
                controller_object.name,
            ),
            |mut acc, param| {
                write!(
                    acc,
                    "\n - {}(old value: {}, new value: {})",
                    param.0, param.1, param.2
                )
                .unwrap();
                acc
            },
        );
        warn!("{}", warning);
    }

    // Check controller validity
    if controller_object.is_valid().is_err() {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Controller {} is not valid", controller_object.name,),
        ));
    };

    // Add controller required parameters to the registry
    let mut controller_added_parameters = ParameterList::new();
    if controller_object.parameters.is_empty() && controller_object.required_parameters.is_empty() {
        warn!(
            "No parameters found for controller {}",
            controller_object.name,
        );
    } else {
        let mut filtered_parameters = ParameterList::new();
        for required_param in &controller_object.required_parameters {
            if let Some(param_value) = controller_object.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = cell_parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = fabric_object.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Required parameter {} not found for controller {}",
                        required_param, controller_object.name,
                    ),
                ));
            }
        }
        controller_object.parameters = filtered_parameters.clone();
        match parameter_list.add_parameters(&filtered_parameters) {
            Ok(added_params) => {
                controller_added_parameters = added_params;
            }
            Err(e) => {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Error with controller parameters for controller {}: ({})",
                        controller_object.name, e
                    ),
                ));
            }
        }
    }

    // generate a hash based on the required parameters for the controller
    let controller_hash = controller_object.get_fingerprint();

    // Check if two controllers with same parameters have the same hash
    if let std::collections::hash_map::Entry::Vacant(e) =
        implemented_controllers.entry(controller_hash.clone())
    {
        e.insert(controller_object.clone());
    } else {
        controller_object.already_defined = true;
    }

    debug!(
        "Controller: {} (hash: {})",
        controller_object.name, controller_hash,
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
    parameter_list
        .remove_parameters(&controller_added_parameters)
        .expect("Error removing controller parameters from registry");

    Ok(())
}

pub fn process_resource(
    resource_object: &mut Resource,
    resource_pool: &serde_json::Value,
    cell_parameters: &ParameterList,
    fabric_object: &Fabric,
    rtl_output_dir: &PathBuf,
    parameter_list: &mut ParameterList,
    implemented_resources: &mut HashMap<String, Resource>,
    current_slot: &mut u64,
) -> Result<(), Error> {
    let mut overwritten_params = Vec::new();

    // Get the resource from the resource pool
    if let Some(resource_entry) = resource_pool
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"].as_str().unwrap() == resource_object.name)
    {
        overwritten_params.extend(
            resource_object
                .update_from_json(&resource_entry.to_string(), false)
                .expect("Failed to update resource from JSON entry"),
        );
    }

    let Some(resource_kind) = resource_object.kind.as_ref() else {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Resource kind not found for resource {}",
                resource_object.name,
            ),
        ));
    };

    let lib_resource_arch = get_arch_from_library(resource_kind, None).map_err(|e| {
        Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Resource {} not found in library (kind: {}): {}",
                resource_object.name, resource_kind, e
            ),
        )
    })?;
    overwritten_params.extend(
        resource_object
            .update_from_json(&lib_resource_arch.to_string(), false)
            .expect("Failed to update resource from JSON entry"),
    );

    if resource_object.slot.is_none() {
        resource_object.slot = Some(*current_slot);
        let resource_object_slot = resource_object.slot.unwrap();
        *current_slot += resource_object.size.unwrap();
        assert_ne!(resource_object_slot, *current_slot); // size cannot be 0
    }

    if !overwritten_params.is_empty() {
        let warning = overwritten_params.iter().fold(
            format!(
                "Some parameters from resource {} (slot {}) were overwritten:",
                resource_object.name,
                resource_object.slot.unwrap(),
            ),
            |mut acc, param| {
                write!(
                    acc,
                    "\n - {}(old value: {}, new value: {})",
                    param.0, param.1, param.2
                )
                .unwrap();
                acc
            },
        );
        warn!("{}", warning);
    }

    // Check resource validity
    if resource_object.is_valid().is_err() {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Resource {} (slot {}) is not valid",
                resource_object.name,
                resource_object.slot.unwrap(),
            ),
        ));
    };

    // Add required parameters to the registry
    let mut resource_added_parameters = ParameterList::new();
    if resource_object.required_parameters.is_empty() {
        warn!(
            "No parameters found for resource {} (slot {})",
            resource_object.name,
            resource_object.slot.unwrap(),
        );
    } else {
        // Filter parameters that are not required
        let mut filtered_parameters = ParameterList::new();
        for required_param in &resource_object.required_parameters {
            if let Some(param_value) = resource_object.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = cell_parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = fabric_object.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Required parameter {} not found for resource {} (slot {})",
                        required_param,
                        resource_object.name,
                        resource_object.slot.unwrap(),
                    ),
                ));
            }
        }
        resource_object.parameters = filtered_parameters.clone();
        match parameter_list.add_parameters(&filtered_parameters) {
            Ok(added_params) => {
                resource_added_parameters = added_params;
            }
            Err(e) => {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Error with resource parameters for resource ({} , slot {}): ({})",
                        resource_object.name,
                        resource_object.slot.unwrap(),
                        e
                    ),
                ));
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
        "Resource: {} (slot: {}): (hash: {})",
        resource_object.name,
        resource_object.slot.unwrap(),
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
    parameter_list
        .remove_parameters(&resource_added_parameters)
        .expect("Error removing resource parameters from registry");

    Ok(())
}

pub fn process_resources(
    cell_parameters: &ParameterList,
    resources_object: &mut Vec<Resource>,
    resource_pool: &serde_json::Value,
    fabric_object: &Fabric,
    rtl_output_dir: &PathBuf,
    parameter_list: &mut ParameterList,
    implemented_resources: &mut HashMap<String, Resource>,
) -> Result<(), Error> {
    // RESOURCES
    let mut current_slot = 0;
    for resource_object in resources_object {
        process_resource(
            resource_object,
            resource_pool,
            cell_parameters,
            fabric_object,
            rtl_output_dir,
            parameter_list,
            implemented_resources,
            &mut current_slot,
        )
        .map_err(|e| {
            Error::new(
                e.kind(),
                format!("While processing resource {}: {}", resource_object.name, e),
            )
        })?;
    }

    Ok(())
}

pub fn process_cell(
    cell_object: &mut Cell,
    pools: &Pools,
    fabric_object: &mut Fabric,
    rtl_output_dir: &PathBuf,
    parameter_list: &mut ParameterList,
    implemented_objects: &mut ImplementedObjects,
) -> Result<(), Error> {
    // Check cell coordinates
    if cell_object.coordinates_list.is_empty() {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Cell {} was declared without coordinates in fabric description",
                cell_object.name
            ),
        ));
    }

    // Get cell parameters from fabric, or cell pool, or library
    let mut overwritten_params = Vec::new();

    let cell_pool = pools.cell_pool;
    if let Some(cell_pool_entry) = cell_pool
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| *entry["name"].as_str().unwrap() == cell_object.name)
    {
        // Update cell parameters from cell pool entry and
        // add overwritten parameters to the list
        overwritten_params.extend(
            cell_object
                .update_from_json(&cell_pool_entry.to_string(), false)
                .expect("Failed to update cell from JSON entry"),
        );
    }

    // Get cell from library if kind is provided
    let Some(cell_kind) = cell_object.kind.as_ref() else {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Cell kind not found for cell {} in fabric",
                cell_object.name,
            ),
        ));
    };
    let Some(lib_cell_arch) = get_arch_from_library(cell_kind, None).ok() else {
        return Err(Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Cell {} (kind: {}) not found in library",
                cell_object.name, cell_kind,
            ),
        ));
    };

    overwritten_params.extend(
        cell_object
            .update_from_json(&lib_cell_arch.to_string(), false)
            .expect("Failed to update cell from JSON entry"),
    );

    // Verify cell validity
    if cell_object.is_valid().is_err() {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Cell {} is not valid", cell_object.name,),
        ));
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
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Required parameter {} not found for cell {}",
                        required_param, cell_object.name,
                    ),
                ));
            }
        }
        cell_object.parameters = filtered_parameters.clone();
        // Add cell parameters to the registry
        match parameter_list.add_parameters(&cell_object.parameters) {
            Ok(added_params) => {
                cell_added_parameters = added_params;
            }
            Err(e) => {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Error with cell parameters for cell {}: ({})",
                        cell_object.name, e
                    ),
                ));
            }
        }
    }

    let cell_name = cell_object.name.clone();
    let cell_parameters = cell_object.parameters.clone();

    let Some(controller_object) = cell_object.controller.as_mut() else {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Cell {} was declared without controller in fabric description",
                cell_name,
            ),
        ));
    };

    process_controller(
        &cell_parameters,
        controller_object,
        pools.controller_pool,
        fabric_object,
        rtl_output_dir,
        parameter_list,
        implemented_objects.implemented_controllers,
    )
    .map_err(|e| {
        Error::new(
            e.kind(),
            format!(
                "While processing controller for cell '{}': {}",
                cell_name, e
            ),
        )
    })?;

    let Some(resources_object) = cell_object.resources.as_mut() else {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Cell {} was declared without resources in fabric description",
                cell_name,
            ),
        ));
    };
    process_resources(
        &cell_parameters,
        resources_object,
        pools.resource_pool,
        fabric_object,
        rtl_output_dir,
        parameter_list,
        implemented_objects.implemented_resources,
    )
    .map_err(|e| {
        Error::new(
            e.kind(),
            format!("While processing resources for cell '{}': {}", cell_name, e),
        )
    })?;

    let cell_hash = cell_object.get_fingerprint();

    let implemented_cells = &mut implemented_objects.implemented_cells;

    // Check if two cells with same parameters have the same hash
    if !implemented_cells.contains_key(&cell_hash) {
        implemented_cells.insert(cell_hash.clone(), cell_object.clone());
    } else {
        cell_object.already_defined = true;
    }

    // Add the cell to the fabric at the different coordinates
    for (row, col) in cell_object.coordinates_list.iter() {
        fabric_object.add_cell(cell_object, *row, *col);
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
    parameter_list
        .remove_parameters(&cell_added_parameters)
        .expect("Error removing cell parameters from registry");

    Ok(())
}

pub fn process_cells(
    cell_list: &serde_json::Value,
    pools: &Pools,
    fabric_object: &mut Fabric,
    rtl_output_dir: &PathBuf,
    parameter_list: &mut ParameterList,
    implemented_objects: &mut ImplementedObjects,
) -> Result<(), Error> {
    // CELLS
    for cell in cell_list.as_array().unwrap() {
        let mut cell_object = match Cell::from_json(&cell.to_string()) {
            Ok(cell) => cell,
            Err(DRRAError::ComponentWithoutNameOrKind) => {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Cell without name or kind found in fabric description: {}",
                        cell
                    ),
                ));
            }
            Err(e) => {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Error parsing cell in fabric description: {}", e),
                ));
            }
        };
        process_cell(
            &mut cell_object,
            pools,
            fabric_object,
            rtl_output_dir,
            parameter_list,
            implemented_objects,
        )?;
    }

    Ok(())
}

pub fn parse_fabric_json(fabric_filepath: &Path) -> Result<Fabric, Error> {
    let fabric_file = match fs::File::open(fabric_filepath) {
        Ok(file) => file,
        Err(err) => {
            return Err(err);
        }
    };
    let fabric_json: serde_json::Value = serde_json::from_reader(fabric_file)?;

    // get the fabric object from the json file
    let fabric = fabric_json
        .get("fabric")
        .expect("Fabric not found in .json");

    // create the fabric object
    let mut fabric_object = Fabric::new();
    fabric_object.add_parameters(get_parameters(
        fabric,
        Some("custom_properties".to_string()),
    ));

    Ok(fabric_object)
}

pub fn gen_rtl(
    fabric_filepath: &Path,
    build_dir: &String,
    output_json: Option<&Path>,
) -> Result<(), Error> {
    let rtl_output_dir = get_rtl_output_dir(build_dir)?;
    let mut parameter_list = ParameterList::new();

    // Create lists for implemented cells, resources and controllers
    let mut implemented_cells: HashMap<String, Cell> = HashMap::new();
    let mut implemented_resources: HashMap<String, Resource> = HashMap::new();
    let mut implemented_controllers: HashMap<String, Controller> = HashMap::new();

    let mut implemented_objects = ImplementedObjects {
        implemented_cells: &mut implemented_cells,
        implemented_resources: &mut implemented_resources,
        implemented_controllers: &mut implemented_controllers,
    };

    // parse the arguments to find the fabric.json input file
    let fabric_filepath = Path::new(fabric_filepath);
    let fabric_file = fs::File::open(fabric_filepath)?;
    let fabric_json: serde_json::Value = serde_json::from_reader(fabric_file)?;

    let cell_pool = fabric_json
        .get("cells")
        .expect("Cell pool not found in fabric");
    let resource_pool = fabric_json
        .get("resources")
        .expect("Resource pool not found in fabric");
    let controller_pool = fabric_json
        .get("controllers")
        .expect("Controller pool not found in fabric");

    let component_pools = Pools {
        cell_pool,
        resource_pool,
        controller_pool,
    };

    let fabric = fabric_json
        .get("fabric")
        .expect("Fabric not found in .json");
    let cell_list = fabric.get("cells_list").expect("Cells not found in fabric");

    // Create the fabric object
    let Ok(mut fabric_object) = parse_fabric_json(fabric_filepath) else {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Error parsing fabric JSON file {:?}", fabric_filepath),
        ));
    };

    // add fabric parameters to the registry
    if fabric_object.parameters.is_empty() {
        warn!("No fabric parameters found in {:?}", fabric_filepath);
    } else {
        parameter_list.add_parameters(&fabric_object.parameters)?;
    }

    process_cells(
        cell_list,
        &component_pools,
        &mut fabric_object,
        &rtl_output_dir,
        &mut parameter_list,
        &mut implemented_objects,
    )
    .map_err(|e| {
        Error::new(
            e.kind(),
            format!(
                "While processing cells in fabric file {:?}: {}",
                fabric_filepath, e
            ),
        )
    })?;

    debug!(
        "Serialized fabric: \n{}",
        serde_json::to_string_pretty(&fabric_object).unwrap()
    );

    if output_json.is_some() {
        write_fabric_json(&fabric_object, output_json.unwrap())?;
    }
    generate_fabric_rtl(&fabric_object, &rtl_output_dir)?;
    copy_common_files(&rtl_output_dir)?;
    copy_testbench_files(&rtl_output_dir)?;

    Ok(())
}

pub fn generate_fabric_rtl(fabric: &Fabric, output_dir: &Path) -> Result<(), Error> {
    // Output the fabric RTL
    let output_folder = Path::new(&output_dir).join("fabric");
    // Remove files in the output directory
    if output_folder.exists() {
        fs::remove_dir_all(&output_folder)?;
    }
    let rtl_output_folder = output_folder.join("rtl");

    fabric.generate_rtl(&rtl_output_folder)?;
    fabric.generate_bender(&output_folder).map_err(|e| {
        Error::new(
            std::io::ErrorKind::Other,
            format!("While generating Bender file for fabric: {}", e),
        )
    })?;

    Ok(())
}

pub fn copy_common_files(output_dir: &Path) -> Result<(), Error> {
    // copy .sv files in utils directory to the output directory
    let common_dir = get_path_from_library(&"common".to_string(), None).unwrap();
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
                let output_dir = Path::new(&output_dir).join("common");
                if !output_dir.exists() {
                    fs::create_dir_all(&output_dir)?;
                }
                let output_dir = output_dir.join(path.file_name().unwrap());
                if !output_dir.exists() {
                    fs::create_dir_all(&output_dir)?;
                }
                // Copy bender file
                let bender_yml_output = output_dir.join("Bender.yml");
                debug!("Copying file: {:?} to {:?}", bender_yml, bender_yml_output);
                let comment = "# This file was automatically generated by Vesyla. DO NOT EDIT.\n\n"
                    .to_string();
                let file_content = comment + &fs::read_to_string(&bender_yml)?;
                fs::write(&bender_yml_output, file_content)?;
                let output_dir = output_dir.join("rtl");

                copy_rtl_dir(&rtl_path, &output_dir)?;
                debug!("Copying directory: {:?} to {:?}", rtl_path, output_dir);
            }
        }
    }

    Ok(())
}

pub fn copy_testbench_files(output_dir: &Path) -> Result<(), Error> {
    // Copy the testbench directory to the output directory
    let testbench_dir = get_path_from_library(&"tb".to_string(), None)?;
    let output_dir = Path::new(&output_dir).join("tb");
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir)?;
    }

    copy_rtl_dir(&testbench_dir, &output_dir)?;

    Ok(())
}

pub fn write_fabric_json(fabric: &Fabric, output_json: &Path) -> Result<(), Error> {
    // Output the fabric object to a JSON file
    // create output directory if it does not exist
    let output_dir = output_json.parent().unwrap();
    let fabric_output_file = output_json;
    fs::create_dir_all(output_dir)?;
    let fabric_output = fs::File::create(fabric_output_file)?;
    serde_json::to_writer_pretty(fabric_output, &fabric)?;
    info!(
        "Generated JSON file for fabric at path: {}",
        fabric_output_file.display()
    );

    Ok(())
}
