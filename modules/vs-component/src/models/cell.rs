use crate::isa::InstructionSet;
use crate::models::controller::Controller;
use crate::models::resource::Resource;
use crate::models::types::{DRRAError, ParameterList, RTLComponent};
use crate::utils::{generate_hash, get_isa_from_library, get_path_from_library, merge_parameters};

use log::warn;
use std::{collections::HashMap, fs, path::Path};

use serde::ser::{Serialize, SerializeMap, Serializer};

#[derive(Clone)]
pub struct Cell {
    pub name: String,
    pub fingerprint: Option<String>,
    pub already_defined: bool,
    pub coordinates_list: Vec<(u64, u64)>,
    pub io_input: Option<bool>,
    pub io_output: Option<bool>,
    pub kind: Option<String>,
    pub controller: Option<Controller>,
    pub resources: Option<Vec<Resource>>,
    pub parameters: ParameterList,
    pub required_parameters: Vec<String>,
    pub isa: Option<InstructionSet>,
}

impl Cell {
    pub fn new(name: String, coordinates_list: Vec<(u64, u64)>) -> Self {
        Cell {
            name,
            fingerprint: None,
            already_defined: false,
            coordinates_list,
            io_input: None,
            io_output: None,
            kind: None,
            controller: None,
            resources: None,
            parameters: ParameterList::new(),
            required_parameters: Vec::new(),
            isa: None,
        }
    }

    pub fn is_valid(&self) -> Result<(), DRRAError> {
        if self.name.is_empty() || self.kind.is_none() {
            return Err(DRRAError::ComponentWithoutNameOrKind);
        }
        if self.controller.is_none() {
            return Err(DRRAError::CellWithoutController);
        }
        if self.isa.is_none() {
            return Err(DRRAError::ComponentWithoutISA);
        }
        if self.resources.is_none() || self.resources.as_ref().unwrap().is_empty() {
            return Err(DRRAError::CellWithoutResources);
        }
        Ok(())
    }

    pub fn from_json(json_str: &str) -> Result<Self, DRRAError> {
        let json_value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        // Name
        let name: String;
        let name_option = json_value.get("cell_name"); //].as_str().unwrap().to_string();
        if name_option.is_none() {
            if json_value.get("name").is_some() {
                name = json_value["name"].as_str().unwrap().to_string();
            } else {
                name = "".to_string();
            }
        } else {
            name = name_option.unwrap().as_str().unwrap().to_string();
        }
        // Coordinates
        let coordinates_option = json_value.get("coordinates");
        let mut coordinates_list = Vec::new();
        if let Some(coordinates) = coordinates_option {
            for coordinate in coordinates.as_array().unwrap() {
                let row = coordinate["row"].as_u64().unwrap();
                let col = coordinate["col"].as_u64().unwrap();
                coordinates_list.push((row, col));
            }
        };

        let mut cell = Cell::new(name, coordinates_list);

        // IO input (optional)
        let io_input = json_value.get("io_input");
        if let Some(io_input) = io_input {
            cell.io_input = Some(io_input.as_bool().unwrap());
        }
        // IO output (optional)
        let io_output = json_value.get("io_output");
        if let Some(io_output) = io_output {
            cell.io_output = Some(io_output.as_bool().unwrap());
        }

        // Cell kind (optional)
        let cell_kind = json_value.get("kind");
        if let Some(cell_kind) = cell_kind {
            cell.kind = Some(cell_kind.as_str().unwrap().to_string());
        } else if cell.name.is_empty() {
            return Err(DRRAError::ComponentWithoutNameOrKind);
        }

        // Parameters
        let json_cell_params = json_value.get("custom_properties");
        if let Some(json_cell_params) = json_cell_params {
            for param in json_cell_params.as_array().unwrap() {
                let name = param
                    .get("name")
                    .expect("Parameter name not found")
                    .as_str()
                    .unwrap();
                let value = param
                    .get("value")
                    .expect("Parameter value not found")
                    .as_u64()
                    .unwrap();
                cell.add_parameter(name.to_string(), value);
            }
        }
        // Required parameters (optional)
        let required_parameters = json_value.get("required_parameters");
        if let Some(required_parameters) = required_parameters {
            for required_parameter in required_parameters.as_array().unwrap() {
                cell.required_parameters
                    .push(required_parameter.as_str().unwrap().to_string());
            }
        }
        // Controller (optional)
        let controller = json_value.get("controller");
        if let Some(controller) = controller {
            let controller_result = Controller::from_json(controller.to_string().as_str());
            if let Ok(controller) = controller_result {
                cell.controller = Some(controller);
            } else {
                warn!("DRRAError parsing controller: {:?}", controller.to_string());
            }
        }
        // Resources (optional)
        let resources = json_value.get("resource_list");
        if let Some(resources) = resources {
            let mut resources_vec = Vec::new();
            for resource in resources.as_array().unwrap() {
                let resource_result = Resource::from_json(resource.to_string().as_str());
                if let Ok(resource_result) = resource_result {
                    resources_vec.push(resource_result);
                } else {
                    warn!("DRRAError parsing resource: {:?}", resource.to_string());
                }
            }
            cell.resources = Some(resources_vec);
        }
        // ISA (optional)
        let isa = json_value.get("isa");
        if let Some(isa) = isa {
            let isa_result =
                InstructionSet::from_json(serde_json::from_str(isa.to_string().as_str()).unwrap());
            if let Ok(isa) = isa_result {
                cell.isa = Some(isa);
            } else {
                panic!("DRRAError parsing ISA: {:?}", isa.to_string());
            }
        } else if let Some(kind) = &cell.kind {
            let lib_isa_json = get_isa_from_library(&kind.clone(), None).unwrap();
            let isa = InstructionSet::from_json(lib_isa_json);
            if isa.is_err() {
                panic!(
                    "DRRAError with ISA for cell {} (kind: {}) cannot be found in library -> {}",
                    &cell.name,
                    kind,
                    isa.err().unwrap()
                );
            } else {
                cell.isa = Some(isa.unwrap());
            }
        }
        Ok(cell)
    }

    pub fn add_parameter(&mut self, name: String, value: u64) {
        self.parameters.insert(name, value);
    }

    pub fn update_from_json(
        &mut self,
        json_str: &str,
        overwrite: bool,
    ) -> Result<Vec<(String, u64, u64)>, DRRAError> {
        if let Ok(cell_from_pool) = Cell::from_json(json_str) {
            // If cell controller was not provided in "fabric" get from cell_from_pool
            if (cell_from_pool.controller.is_some() && self.controller.is_none()) || overwrite {
                self.controller = cell_from_pool.controller.clone();
            }
            // If cell resources were not provided in "fabric" get from cell_from_pool
            if (cell_from_pool.resources.is_some() && self.resources.is_none()) || overwrite {
                self.resources = cell_from_pool.resources.clone();
            }
            // Check io input
            if (cell_from_pool.io_input.is_some() && self.io_input.is_none()) || overwrite {
                self.io_input = cell_from_pool.io_input;
            }
            // Check io output
            if (cell_from_pool.io_output.is_some() && self.io_output.is_none()) || overwrite {
                self.io_output = cell_from_pool.io_output;
            }
            // If cell kind was not provided in "fabric" get from cell_from_pool
            if (cell_from_pool.kind.is_some() && self.kind.is_none()) || overwrite {
                self.kind = cell_from_pool.kind.clone();
            }

            // append parameters from cell_from_pool to cell required parameters
            for param_key in cell_from_pool.parameters.keys() {
                if !self.required_parameters.contains(param_key) {
                    self.required_parameters.push(param_key.clone());
                }
            }

            // append required parameters from cell_from_pool to cell
            if (!cell_from_pool.required_parameters.is_empty()) || overwrite {
                self.required_parameters
                    .extend(cell_from_pool.required_parameters);
            }
            // get isa from cell_from_pool if is not already defined
            if (cell_from_pool.isa.is_some() && self.isa.is_none()) || overwrite {
                self.isa = cell_from_pool.isa;
            }
            // append parameters from cell_from_pool to cell_parameters
            let overwritten_params =
                merge_parameters(&mut self.parameters, &cell_from_pool.parameters)?;
            return Ok(overwritten_params);
        }
        Err(DRRAError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Not implemented",
        )))
    }

    fn get_fingerprint_table(&self) -> HashMap<String, String> {
        let mut fingerprint_table = HashMap::new();
        if let Some(controller) = &self.controller {
            if let Some(fingerprint) = &controller.fingerprint {
                if !fingerprint_table.contains_key(&controller.name) {
                    fingerprint_table.insert(controller.name.clone(), fingerprint.clone());
                }
            }
        }
        if let Some(resources) = &self.resources {
            for resource in resources.iter() {
                if let Some(fingerprint) = &resource.fingerprint {
                    if !fingerprint_table.contains_key(&resource.name) {
                        fingerprint_table.insert(resource.name.clone(), fingerprint.clone());
                    }
                }
            }
        }
        fingerprint_table
    }
}

impl RTLComponent for Cell {
    fn generate_bender(&self, output_folder: &Path) -> Result<(), DRRAError> {
        let component_path = get_path_from_library(self.kind.as_ref().unwrap(), None).unwrap();
        let bender_filepath = Path::new(&component_path).join("Bender.yml");

        if !bender_filepath.exists() {
            panic!(
                "Bender file not found for component: {:?} (looking for {:?})",
                self.kind.as_ref().unwrap(),
                bender_filepath
            );
        }

        // Read the bender file
        let mut bender_file =
            serde_yml::from_str::<serde_yml::Value>(&fs::read_to_string(&bender_filepath).unwrap())
                .unwrap();

        // Replace package["name"] with the cell name with the hash
        let cell_with_hash: String = format!("{}_{}", self.name, self.clone().get_fingerprint());
        bender_file["package"]["name"] = serde_yml::Value::String(cell_with_hash);

        // Add controller and all resources to the dependencies
        let mut dependencies = serde_yml::Mapping::new();
        if let Some(controller) = &self.controller {
            let controller_with_hash: String = format!(
                "{}_{}",
                controller.name,
                controller.clone().get_fingerprint()
            );
            let mut controller_path_map = serde_yml::Mapping::new();
            controller_path_map.insert(
                serde_yml::Value::String("path".to_string()),
                serde_yml::Value::String(format!("../../controllers/{}", controller_with_hash)),
            );
            dependencies.insert(
                serde_yml::Value::String(controller_with_hash),
                serde_yml::Value::Mapping(controller_path_map),
            );
        }
        if let Some(resources) = &self.resources {
            for resource in resources.iter() {
                let resource_with_hash: String =
                    format!("{}_{}", resource.name, resource.clone().get_fingerprint());
                let mut resource_path_map = serde_yml::Mapping::new();
                resource_path_map.insert(
                    serde_yml::Value::String("path".to_string()),
                    serde_yml::Value::String(format!("../../resources/{}", resource_with_hash)),
                );
                dependencies.insert(
                    serde_yml::Value::String(resource_with_hash),
                    serde_yml::Value::Mapping(resource_path_map),
                );
            }
        }

        bender_file["dependencies"] = serde_yml::Value::Mapping(dependencies.clone());

        // Write the bender file
        let bender_file_path = output_folder.join("Bender.yml");
        let comment =
            "# This file was automatically generated by Vesyla. DO NOT EDIT.\n\n".to_string();
        fs::write(
            bender_file_path,
            comment + &serde_yml::to_string(&bender_file).unwrap(),
        )?;

        Ok(())
    }

    fn generate_hash(&mut self) -> String {
        let controller_fingerprint = self
            .controller
            .as_ref()
            .unwrap()
            .fingerprint
            .clone()
            .unwrap();
        let resources_fingerprints = self
            .resources
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.fingerprint.clone().unwrap())
            .collect::<Vec<String>>();
        let mut fingerprints = vec![controller_fingerprint];
        fingerprints.extend(resources_fingerprints);
        self.fingerprint = Some(generate_hash(fingerprints, &self.parameters));
        self.fingerprint.clone().unwrap()
    }

    fn get_fingerprint(&mut self) -> String {
        if self.fingerprint.is_none() {
            self.generate_hash()
        } else {
            self.fingerprint.clone().unwrap()
        }
    }

    fn kind(&self) -> &str {
        self.kind.as_ref().unwrap()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Serialize for Cell {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize the struct as a map with the following keys:
        // - cell_name
        // - kind
        // - controller
        // - resource_list
        // - custom_properties
        // - required_parameters
        let mut state = serializer.serialize_map(Some(7))?;
        state.serialize_entry("name", &self.name)?;
        if let Some(fingerprint) = &self.fingerprint {
            state.serialize_entry("fingerprint", fingerprint)?;
        }
        state.serialize_entry("already_defined", &self.already_defined)?;
        //state.serialize_entry("coordinates", &self.coordinates_list)?;
        if let Some(kind) = &self.kind {
            state.serialize_entry("kind", kind)?;
        }
        if let Some(io_input) = self.io_input {
            state.serialize_entry("io_input", &io_input)?;
        }
        if let Some(io_output) = self.io_output {
            state.serialize_entry("io_output", &io_output)?;
        }
        if !self.parameters.is_empty() {
            state.serialize_entry("parameters", &self.parameters)?;
        }
        if let Some(controller) = &self.controller {
            state.serialize_entry("controller", controller)?;
        }
        if let Some(resources) = &self.resources {
            state.serialize_entry("resources_list", resources)?;
            let mut used_slots: Vec<u64> = Vec::new();
            let controller_size = if let Some(controller) = &self.controller {
                controller.size.unwrap_or(0)
            } else {
                0
            };
            let mut unused_slots: Vec<u64> = (0..controller_size).collect();
            for resource in resources.iter() {
                let slot = resource.slot;
                let size = resource.size;
                if slot.is_some() && size.is_some() {
                    let slot = slot.unwrap();
                    let size = size.unwrap();
                    for slot_number in slot..(slot + size) {
                        used_slots.push(slot_number);
                        unused_slots.retain(|&x| x != slot_number);
                    }
                }
            }
            state.serialize_entry("used_slots", &used_slots)?;
            state.serialize_entry("unused_slots", &unused_slots)?;
        }
        if let Some(isa) = &self.isa {
            state.serialize_entry("isa", isa)?;
        }
        state.serialize_entry("fingerprint_table", &self.get_fingerprint_table())?;
        state.end()
    }
}

pub struct CellWithCoordinates {
    pub cell: Cell,
    pub coordinates: (u64, u64),
}

impl Serialize for CellWithCoordinates {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize the struct as a map with the following keys:
        // - cell
        // - coordinates
        let mut state = serializer.serialize_map(Some(2))?;
        let row = self.coordinates.0;
        let col = self.coordinates.1;
        let coordinates_hashmap = [("row", &row), ("col", &col)]
            .iter()
            .cloned()
            .collect::<HashMap<&str, &u64>>();
        state.serialize_entry("coordinates", &coordinates_hashmap)?;
        state.serialize_entry("cell", &self.cell)?;
        state.end()
    }
}
