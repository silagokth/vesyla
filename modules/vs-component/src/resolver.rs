use crate::models::{
    alimp::Alimp,
    cell::Cell,
    controller::Controller,
    drra::Fabric,
    isa::ComponentInstructionSet,
    resource::Resource,
    types::{DRRAError, ParameterList},
};

use crate::utils::{get_arch_from_library, get_isa_from_library, get_parameters};

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
        if pool.is_null() {
            return Ok(());
        }

        for entry in pool.as_array().unwrap() {
            let mut controller = Controller::from_json(&entry.to_string())?;

            if let Some(kind) = controller.kind.clone() {
                let lib_data = self.get_library_data(&kind)?;
                let _ = controller.update_from_json(&lib_data.to_string(), false);
                let controller_isa = get_isa_from_library(&kind, None)?;
                let controller_isa = ComponentInstructionSet::from_json(controller_isa)?;
                controller.isa = Some(controller_isa);
            } else {
                panic!("Controller kind is required");
            }

            self.controller_templates
                .insert(controller.name.clone(), controller);
        }

        Ok(())
    }

    fn build_resource_templates(&mut self, pool: &serde_json::Value) -> Result<(), DRRAError> {
        if pool.is_null() {
            return Ok(());
        }

        for entry in pool.as_array().unwrap() {
            let mut resource = Resource::from_json(&entry.to_string())?;

            if let Some(kind) = resource.kind.clone() {
                let lib_data = self.get_library_data(&kind)?;
                let _ = resource.update_from_json(&lib_data.to_string(), false)?;
                let resource_isa = get_isa_from_library(&kind, None)?;
                let resource_isa = ComponentInstructionSet::from_json(resource_isa)?;
                resource.isa = Some(resource_isa);
            }

            self.resource_templates
                .insert(resource.name.clone(), resource);
        }
        Ok(())
    }

    fn build_cell_templates(&mut self, pool: &serde_json::Value) -> Result<(), DRRAError> {
        if pool.is_null() {
            log::warn!("No cell templates found in input file.");
            return Ok(());
        }

        for entry in pool.as_array().unwrap() {
            log::info!("Processing cell template entry: {}", entry);
            let mut cell = Cell::from_json(&entry.to_string())?;

            if let Some(kind) = &cell.kind {
                let lib_data = self.get_library_data(kind)?;
                log::info!("Library data for cell '{}': {}", cell.name, lib_data);
                let _ = cell.update_from_json(&lib_data.to_string(), false)?;
            }

            log::info!("Loaded cell template '{}'", cell.name);
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
        log::info!("Library data for '{}': {}", name, lib_data);
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
            let cell_from_list = Cell::from_json(&cell_placement.to_string())?;
            let mut cell_kind = cell_from_list.kind.as_ref();
            let cell_from_pool = self
                .cell_templates
                .get(&cell_from_list.name)
                .ok_or_else(|| {
                    Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Cell '{}' not found in templates", cell_from_list.name),
                    )
                })?
                .clone();
            if cell_kind.is_none() {
                cell_kind = cell_from_pool.kind.as_ref();
            }
            let cell_from_lib_json = get_arch_from_library(cell_kind.unwrap(), None)?;
            let cell_from_lib = Cell::from_json(&cell_from_lib_json.to_string())?;
            let resolved_cell = self
                .resolve_cell(&cell_from_list, &cell_from_pool, &cell_from_lib, fabric)
                .map_err(|e| {
                    DRRAError::Io(Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "Failed to resolve cell '{}': {}",
                            cell_from_list.name.clone(),
                            e
                        ),
                    ))
                })?;
            resolved_cells.push(resolved_cell);
        }

        Ok(resolved_cells)
    }

    fn resolve_cell(
        &self,
        cell_from_list: &Cell,
        cell_from_pool: &Cell,
        cell_from_lib: &Cell,
        fabric: &Fabric,
    ) -> Result<Cell, DRRAError> {
        // start with the cell from the library as the base
        let mut resolved_cell = cell_from_lib.clone();
        resolved_cell.overwrite(cell_from_pool)?;
        resolved_cell.overwrite(cell_from_list)?;

        self.resolve_cell_parameters(&mut resolved_cell, fabric)?;

        if let Some(controller) = resolved_cell.controller {
            let controller_from_list = controller.clone();
            let controller_name = &controller.name;
            let mut controller_kind = controller.kind.as_ref();
            let controller_from_pool =
                self.controller_templates
                    .get(controller_name)
                    .ok_or_else(|| {
                        Error::new(
                            std::io::ErrorKind::NotFound,
                            format!("Controller '{}' not found in templates", controller_name),
                        )
                    })?;
            if controller_kind.is_none() {
                controller_kind = controller_from_pool.kind.as_ref();
            }
            let controller_from_lib_json = get_arch_from_library(controller_kind.unwrap(), None)?;
            let controller_from_lib = Controller::from_json(&controller_from_lib_json.to_string())?;
            let resolved_controller = self
                .resolve_controller(
                    &controller_from_list,
                    controller_from_pool,
                    &controller_from_lib,
                    &resolved_cell.parameters,
                    fabric,
                )
                .map_err(|e| {
                    DRRAError::Io(Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Failed to resolve controller '{}': {}", controller_name, e),
                    ))
                })?;
            resolved_cell.controller = Some(resolved_controller);
        } else {
            return Err(DRRAError::CellWithoutController);
        }

        // Resolve resources
        if let Some(resource_list) = resolved_cell.resources {
            let resolved_resources =
                self.resolve_resources(&resource_list, &resolved_cell.parameters, fabric)?;
            resolved_cell.resources = Some(resolved_resources);
        } else {
            return Err(DRRAError::CellWithoutResources);
        }

        // Compose ISA
        resolved_cell.get_isa()?;

        Ok(resolved_cell)
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
        controller_from_list: &Controller,
        controller_from_pool: &Controller,
        controller_from_lib: &Controller,
        cell_parameters: &ParameterList,
        fabric: &Fabric,
    ) -> Result<Controller, DRRAError> {
        let mut resolved_controller = controller_from_lib.clone();
        resolved_controller.overwrite(controller_from_pool)?;
        resolved_controller.overwrite(controller_from_list)?;

        // Resolve controller parameters with hierarchy: controller -> cell -> fabric
        let mut filtered_parameters = ParameterList::new();

        for required_param in &resolved_controller.required_parameters {
            if let Some(param_value) = resolved_controller.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = cell_parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = fabric.parameters.get(required_param) {
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else {
                return Err(DRRAError::ParameterNotFound(required_param.clone()));
            }
        }
        resolved_controller.parameters = filtered_parameters;

        resolved_controller.validate()?;

        Ok(resolved_controller)
    }

    fn resolve_resources(
        &self,
        resource_list: &Vec<Resource>,
        cell_parameters: &ParameterList,
        fabric: &Fabric,
    ) -> Result<Vec<Resource>, DRRAError> {
        let mut resources = Vec::new();
        let mut current_slot = 0;

        for resource in resource_list {
            let resource_from_list = resource.clone();
            let resource_name = &resource.name;
            let mut resource_kind = resource.kind.as_ref();
            let resource_from_pool =
                self.resource_templates.get(&resource.name).ok_or_else(|| {
                    Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Resource '{}' not found in templates", resource_name),
                    )
                })?;
            if resource_kind.is_none() {
                resource_kind = resource_from_pool.kind.as_ref();
            }
            let resource_from_lib_json = get_arch_from_library(resource_kind.unwrap(), None)?;
            let resource_from_lib = Resource::from_json(&resource_from_lib_json.to_string())?;
            let mut resolved_resource = self
                .resolve_resource(
                    &resource_from_list,
                    resource_from_pool,
                    &resource_from_lib,
                    cell_parameters,
                    fabric,
                )
                .map_err(|e| {
                    DRRAError::Io(Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Failed to resolve resource '{}': {}", resource_name, e),
                    ))
                })?;

            if resource.slot.is_none() {
                resolved_resource.slot = Some(current_slot);
                current_slot += resolved_resource.size.unwrap_or(1);
            }

            resolved_resource.validate()?;

            resources.push(resolved_resource);
        }

        Ok(resources)
    }

    fn resolve_resource(
        &self,
        resource_from_list: &Resource,
        resource_from_pool: &Resource,
        resource_from_lib: &Resource,
        cell_parameters: &ParameterList,
        fabric: &Fabric,
    ) -> Result<Resource, DRRAError> {
        let mut resolved_resource = resource_from_lib.clone();
        resolved_resource.overwrite(resource_from_pool)?;
        resolved_resource.overwrite(resource_from_list)?;

        log::debug!(
            "Resolving resource '{}', kind: '{:?}'",
            resolved_resource.name,
            resolved_resource.kind
        );

        // Resolve resource parameters with hierarchy
        let mut filtered_parameters = ParameterList::new();

        for required_param in &resolved_resource.required_parameters {
            if let Some(param_value) = resolved_resource.parameters.get(required_param) {
                // Resource's own parameter (highest priority)
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = cell_parameters.get(required_param) {
                // Cell parameter (medium priority)
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else if let Some(param_value) = fabric.parameters.get(required_param) {
                // Fabric parameter (lowest priority)
                filtered_parameters.insert(required_param.clone(), *param_value);
            } else {
                return Err(DRRAError::ParameterNotFound(required_param.clone()));
            }
        }
        resolved_resource.parameters = filtered_parameters;

        Ok(resolved_resource)
    }
}
