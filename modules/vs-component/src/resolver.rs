use crate::models::{
    alimp::Alimp,
    cell::Cell,
    controller::Controller,
    drra::Fabric,
    resource::Resource,
    types::{DRRAError, ParameterList},
};
use crate::utils::*;

use std::{collections::HashMap, fs, io::Error, path::Path};

pub struct ResolvedAlimp {
    pub alimp: Alimp,
    pub resolved_cells: Vec<Cell>,
}

pub struct HierarchicalResolver {
    library_cache: HashMap<String, serde_json::Value>,
    cell_templates: HashMap<String, Cell>,
    controller_templates: HashMap<String, Controller>,
    resource_templates: HashMap<String, Resource>,
}

impl HierarchicalResolver {
    pub fn new() -> Self {
        Self {
            library_cache: HashMap::new(),
            controller_templates: HashMap::new(),
            resource_templates: HashMap::new(),
            cell_templates: HashMap::new(),
        }
    }
    pub fn resolve_alimp(&mut self, alimp_filepath: &Path) -> Result<ResolvedAlimp, DRRAError> {
        let input: serde_json::Value = serde_json::from_reader(fs::File::open(alimp_filepath)?)
            .map_err(|e| DRRAError::Io(Error::new(std::io::ErrorKind::InvalidInput, e)))?;

        self.build_controller_templates(&input["controllers"])?;
        self.build_resource_templates(&input["resources"])?;
        self.build_cell_templates(&input["cells"])?;

        let mut alimp = Alimp::new();
        alimp.drra = self.build_base_fabric(&input["fabric"]).ok();

        let resolved_cells =
            self.resolve_cells(&input["fabric"]["cells_list"], alimp.drra.as_mut().unwrap())?;

        for cell in &resolved_cells {
            for (row, col) in &cell.coordinates_list {
                alimp.drra.as_mut().unwrap().add_cell(cell, *row, *col);
            }
        }

        alimp.drra.as_mut().unwrap().generate_fingerprints()?;
        alimp.validate()?;

        Ok(ResolvedAlimp {
            alimp,
            resolved_cells,
        })
    }

    fn build_base_fabric(&self, fabric_json: &serde_json::Value) -> Result<Fabric, DRRAError> {
        let mut fabric = Fabric::new();
        fabric.add_parameters(get_parameters(
            fabric_json,
            Some("custom_properties".to_string()),
        ));
        Ok(fabric)
    }

    fn build_controller_templates(&mut self, pool: &serde_json::Value) -> Result<(), DRRAError> {
        for entry in pool.as_array().unwrap() {
            let mut controller = Controller::from_json(&entry.to_string())?;

            if let Some(kind) = &controller.kind {
                let lib_data = self.get_library_data(kind)?;
                let _ = &controller.update_from_json(&lib_data.to_string(), false)?;
            } else {
                panic!("Controller kind is required");
            }

            self.controller_templates
                .insert(controller.name.clone(), controller);
        }

        Ok(())
    }

    fn build_resource_templates(&mut self, pool: &serde_json::Value) -> Result<(), DRRAError> {
        for entry in pool.as_array().unwrap() {
            let mut resource = Resource::from_json(&entry.to_string())?;

            if let Some(kind) = &resource.kind {
                let lib_data = self.get_library_data(kind)?;
                let _ = resource.update_from_json(&lib_data.to_string(), false)?;
            }

            self.resource_templates
                .insert(resource.name.clone(), resource);
        }
        Ok(())
    }

    fn build_cell_templates(&mut self, pool: &serde_json::Value) -> Result<(), DRRAError> {
        for entry in pool.as_array().unwrap() {
            let mut cell = Cell::from_json(&entry.to_string())?;

            if let Some(kind) = &cell.kind {
                let lib_data = self.get_library_data(kind)?;
                let _ = cell.update_from_json(&lib_data.to_string(), false)?;
            }

            self.cell_templates.insert(cell.name.clone(), cell);
        }
        Ok(())
    }

    fn get_library_data(&mut self, name: &String) -> Result<serde_json::Value, DRRAError> {
        if let Some(cached) = self.library_cache.get(name) {
            return Ok(cached.clone());
        }

        log::info!("Loading library data for '{}'", name);
        let lib_data = get_arch_from_library(name, None)?;
        self.library_cache.insert(name.clone(), lib_data.clone());
        Ok(lib_data)
    }

    fn resolve_cells(
        &self,
        cells_list: &serde_json::Value,
        fabric: &Fabric,
    ) -> Result<Vec<Cell>, DRRAError> {
        let mut resolved_cells = Vec::new();

        for cell_placement in cells_list.as_array().unwrap() {
            let resolved_cell = self.resolve_cell(cell_placement, fabric)?;
            resolved_cells.push(resolved_cell);
        }

        Ok(resolved_cells)
    }

    fn resolve_cell(
        &self,
        cell_placement: &serde_json::Value,
        fabric: &Fabric,
    ) -> Result<Cell, DRRAError> {
        let cell_name = cell_placement["cell_name"].as_str().unwrap();
        let mut cell = self
            .cell_templates
            .get(cell_name)
            .ok_or_else(|| {
                Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Cell template '{}' not found", cell_name),
                )
            })?
            .clone();

        // add coordinates from placement
        cell.coordinates_list = cell_placement["coordinates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|coord| {
                (
                    coord["row"].as_u64().unwrap(),
                    coord["col"].as_u64().unwrap(),
                )
            })
            .collect();

        self.resolve_cell_parameters(&mut cell, fabric)?;

        if let Some(controller) = cell.controller {
            let resolved_controller =
                self.resolve_controller(&controller, &cell.parameters, fabric)?;
            println!(
                "Resolved controller '{:?}' for cell '{}'",
                resolved_controller.kind.clone(),
                cell.name
            );
            cell.controller = Some(resolved_controller);
        } else {
            return Err(DRRAError::CellWithoutController);
        }

        // Resolve resources
        if let Some(resource_list) = cell.resources {
            let resolved_resources =
                self.resolve_resources(&resource_list, &cell.parameters, fabric)?;
            cell.resources = Some(resolved_resources);
        } else {
            return Err(DRRAError::CellWithoutResources);
        }

        Ok(cell)
    }

    fn resolve_cell_parameters(&self, cell: &mut Cell, fabric: &Fabric) -> Result<(), DRRAError> {
        let mut filtered_parameters = ParameterList::new();

        for required_param in &cell.required_parameters {
            if let Some(param_value) = cell.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = fabric.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else {
                return Err(DRRAError::ParameterNotFound(required_param.clone()));
            }
        }

        cell.parameters = filtered_parameters;
        Ok(())
    }

    fn resolve_controller(
        &self,
        controller: &Controller,
        cell_parameters: &ParameterList,
        fabric: &Fabric,
    ) -> Result<Controller, DRRAError> {
        // Clone the controller template
        let mut controller = self
            .controller_templates
            .get(controller.name.as_str())
            .ok_or_else(|| {
                Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Controller template '{}' not found",
                        controller.name.as_str()
                    ),
                )
            })?
            .clone();

        // Resolve controller parameters with hierarchy: controller -> cell -> fabric
        let mut filtered_parameters = ParameterList::new();

        for required_param in &controller.required_parameters {
            if let Some(param_value) = controller.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = cell_parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = fabric.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else {
                return Err(DRRAError::ParameterNotFound(required_param.clone()));
            }
        }

        controller.parameters = filtered_parameters;
        Ok(controller)
    }

    fn resolve_resources(
        &self,
        resource_list: &Vec<Resource>,
        cell_parameters: &ParameterList,
        fabric: &Fabric,
    ) -> Result<Vec<Resource>, Error> {
        let mut resources = Vec::new();
        let mut current_slot = 0;

        for resource in resource_list {
            let resource_name = resource.name.as_str();

            let mut resource = self
                .resource_templates
                .get(resource_name)
                .ok_or_else(|| {
                    Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Resource template '{}' not found", resource_name),
                    )
                })?
                .clone();

            // Assign slot if not already assigned
            if resource.slot.is_none() {
                resource.slot = Some(current_slot);
                current_slot += resource.size.unwrap_or(1);
            }

            // Resolve resource parameters with hierarchy
            let mut filtered_parameters = ParameterList::new();

            for required_param in &resource.required_parameters {
                if let Some(param_value) = resource.parameters.get(required_param) {
                    // Resource's own parameter (highest priority)
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else if let Some(param_value) = cell_parameters.get(required_param) {
                    // Cell parameter (medium priority)
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else if let Some(param_value) = fabric.parameters.get(required_param) {
                    // Fabric parameter (lowest priority)
                    filtered_parameters.insert(required_param.clone(), *param_value);
                } else {
                    return Err(Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "Required parameter {} not found for resource {} (slot {})",
                            required_param,
                            resource.name,
                            resource.slot.unwrap()
                        ),
                    ));
                }
            }

            resource.parameters = filtered_parameters;
            resources.push(resource);
        }

        Ok(resources)
    }
}
