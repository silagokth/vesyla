use crate::isa::*;
use crate::utils::{
    generate_hash, generate_rtl_for_component, get_isa_from_library, get_path_from_library,
    merge_parameters,
};
use core::panic;
use log::warn;
use serde::ser::{Serialize, SerializeMap, Serializer};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

pub type ParameterList = BTreeMap<String, u64>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    ResourceDeclaredAsController,
    ControllerDeclaredAsResource,
    ComponentWithoutNameOrKind,
    ComponentWithoutISA,
    UnknownComponentType,
    CellWithoutController,
    CellWithoutResources,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ResourceDeclaredAsController => write!(f, "Resource declared as controller"),
            Error::ControllerDeclaredAsResource => write!(f, "Controller declared as resource"),
            Error::ComponentWithoutNameOrKind => write!(f, "Component without name or kind"),
            Error::ComponentWithoutISA => write!(f, "Component without ISA"),
            Error::UnknownComponentType => write!(f, "Unknown component type"),
            Error::Io(err) => write!(f, "IO error: {}", err),
            Error::CellWithoutController => write!(f, "Cell without controller"),
            Error::CellWithoutResources => write!(f, "Cell without resources"),
        }
    }
}

impl std::convert::From<std::io::Error> for Error {
    fn from(_error: std::io::Error) -> Self {
        Error::Io(_error)
    }
}

pub trait RTLComponent {
    fn generate_rtl(&self, output_folder: &Path) -> std::io::Result<()>;
    fn generate_bender(&self, output_folder: &Path) -> Result<(), Error>;
    fn generate_hash(&mut self) -> String;
    fn get_fingerprint(&mut self) -> String;
}

#[derive(Clone)]
pub struct Controller {
    pub name: String,
    pub fingerprint: Option<String>,
    pub kind: Option<String>,
    pub already_defined: bool,
    pub size: Option<u64>,
    pub io_input: Option<bool>,
    pub io_output: Option<bool>,
    pub component_type: String,
    pub parameters: ParameterList,
    pub required_parameters: Vec<String>,
    pub isa: Option<InstructionSet>,
}

impl Controller {
    pub fn new(name: String, size: Option<u64>) -> Self {
        Controller {
            name,
            fingerprint: None,
            kind: None,
            already_defined: false,
            size,
            io_input: None,
            io_output: None,
            component_type: "controller".to_string(),
            parameters: ParameterList::new(),
            required_parameters: Vec::new(),
            isa: None,
        }
    }

    pub fn is_valid(&self) -> Result<(), Error> {
        if self.name.is_empty() || self.kind.is_none() || self.size.is_none() {
            return Err(Error::ComponentWithoutNameOrKind);
        }
        if self.isa.is_none() {
            return Err(Error::ComponentWithoutISA);
        }
        if self.required_parameters.is_empty() {
            warn!("Controller {} has no required parameters", self.name);
        }
        Ok(())
    }

    pub fn from_json(json_str: &str) -> Result<Self, Error> {
        let json_value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        // Name
        if json_value.is_string() {
            let name = json_value.as_str().unwrap().to_string();
            return Ok(Controller::new(name, None));
        }
        let name: String;
        let name_option = json_value.get("controller_name"); //].as_str().unwrap().to_string();
        if name_option.is_none() {
            if json_value.get("name").is_some() {
                name = json_value["name"].as_str().unwrap().to_string();
            } else {
                name = "".to_string();
            }
        } else {
            name = name_option.unwrap().as_str().unwrap().to_string();
        }
        // Size (optional)
        let size = json_value.get("size").map(|x| x.as_u64().unwrap());
        let mut controller = Controller::new(name, size);

        // Kind (optional)
        let controller_kind = json_value.get("kind");
        if let Some(controller_kind) = controller_kind {
            controller.kind = Some(controller_kind.as_str().unwrap().to_string());
        } else if controller.name.is_empty() {
            return Err(Error::ComponentWithoutNameOrKind);
        }

        // IO input (optional)
        let io_input = json_value.get("io_input");
        if let Some(io_input) = io_input {
            controller.io_input = Some(io_input.as_bool().unwrap());
        }
        // IO output (optional)
        let io_output = json_value.get("io_output");
        if let Some(io_output) = io_output {
            controller.io_output = Some(io_output.as_bool().unwrap());
        }

        // Check component type
        let component_type = json_value.get("component_type");
        if let Some(component_type) = component_type {
            if component_type.as_str().unwrap() != controller.component_type
                && !component_type.as_str().unwrap().is_empty()
            {
                if component_type.as_str().unwrap() == "resource" {
                    return Err(Error::ControllerDeclaredAsResource);
                } else {
                    return Err(Error::UnknownComponentType);
                }
            }
        }
        // Parameters (optional)
        let json_controller_params = json_value.get("custom_properties");
        if let Some(json_controller_params) = json_controller_params {
            for param in json_controller_params.as_array().unwrap() {
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
                controller.add_parameter(name.to_string(), value);
            }
        }
        // Required parameters (optional)
        let required_parameters = json_value.get("required_parameters");
        if let Some(required_parameters) = required_parameters {
            for required_parameter in required_parameters.as_array().unwrap() {
                controller
                    .required_parameters
                    .push(required_parameter.as_str().unwrap().to_string());
            }
        }
        // ISA (optional)
        let isa = json_value.get("isa");
        if let Some(isa) = isa {
            let isa_result = InstructionSet::from_json(isa.clone());
            if let Err(isa) = isa_result {
                panic!("Error parsing ISA for controller: {:?}", isa);
            } else {
                controller.isa = Some(isa_result.unwrap());
            }
        } else if let Some(kind) = &controller.kind {
            let lib_isa_json = get_isa_from_library(&kind.clone(), None).unwrap();
            let isa = InstructionSet::from_json(lib_isa_json);
            if isa.is_err() {
                panic!(
                    "Error with ISA for controller {} (kind: {}) cannot be found in library -> {}",
                    &controller.name,
                    kind,
                    isa.err().unwrap()
                );
            } else {
                controller.isa = Some(isa.unwrap());
            }
        }
        Ok(controller)
    }

    pub fn update_from_json(
        &mut self,
        json_str: &str,
        overwrite: bool,
    ) -> Result<Vec<(String, u64, u64)>, Error> {
        if let Ok(incoming_controller) = Controller::from_json(json_str) {
            // Check size
            if (self.size.is_none() && incoming_controller.size.is_some()) || overwrite {
                self.size = incoming_controller.size;
            }
            // Check io input
            if (incoming_controller.io_input.is_some() && self.io_input.is_none()) || overwrite {
                self.io_input = incoming_controller.io_input;
            }
            // Check io output
            if (incoming_controller.io_output.is_some() && self.io_output.is_none()) || overwrite {
                self.io_output = incoming_controller.io_output;
            }

            // append parameters from incoming_controller to controller required parameters
            for param_key in incoming_controller.parameters.keys() {
                if !self.required_parameters.contains(param_key) {
                    self.required_parameters.push(param_key.clone());
                }
            }

            // append required parameters from incoming_controller to controller required parameters
            if !incoming_controller.required_parameters.is_empty() || overwrite {
                self.required_parameters
                    .extend(incoming_controller.required_parameters);
            }

            // check if the controller has kind
            if (incoming_controller.kind.is_some() && self.kind.is_none()) || overwrite {
                self.kind = incoming_controller.kind;
            }
            // get isa from incoming_controller if is not already defined
            if (incoming_controller.isa.is_some() && self.isa.is_none()) || overwrite {
                self.isa = incoming_controller.isa;
            }

            // append parameters from incoming_controller to controller_parameters
            let overwritten_params =
                merge_parameters(&mut self.parameters, &incoming_controller.parameters)?;
            return Ok(overwritten_params);
        }
        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Not implemented",
        )))
    }

    pub fn add_parameter(&mut self, name: String, value: u64) {
        self.parameters.insert(name, value);
    }
}

impl RTLComponent for Controller {
    fn generate_rtl(&self, output_folder: &Path) -> std::io::Result<()> {
        generate_rtl_for_component(
            self.kind.as_ref().unwrap(),
            self.name.as_str(),
            output_folder,
            &self,
        )
    }

    fn generate_bender(&self, output_folder: &Path) -> Result<(), Error> {
        let component_path = get_path_from_library(self.kind.as_ref().unwrap(), None).unwrap();
        let bender_filepath = Path::new(&component_path).join("Bender.yml");
        let component_with_hash: String =
            format!("{}_{}", self.name, self.clone().get_fingerprint());

        // Read the bender file
        let mut bender_file =
            serde_yml::from_str::<serde_yml::Value>(&fs::read_to_string(&bender_filepath).unwrap())
                .unwrap();
        bender_file["package"]["name"] = serde_yml::Value::String(component_with_hash);
        if !bender_filepath.exists() {
            panic!(
                "Bender file not found for component: {:?} (looking for {:?})",
                self.kind.as_ref().unwrap(),
                bender_filepath
            );
        }
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
        self.fingerprint = Some(generate_hash(vec![self.name.clone()], &self.parameters));
        self.fingerprint.clone().unwrap()
    }

    fn get_fingerprint(&mut self) -> String {
        if self.fingerprint.is_none() {
            self.generate_hash()
        } else {
            self.fingerprint.clone().unwrap()
        }
    }
}

impl Serialize for Controller {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize the struct as a map with the following keys:
        // - controller_name
        // - kind
        // - size
        // - io_input
        // - io_output
        // - component_type
        // - custom_properties
        // - required_parameters
        let mut state = serializer.serialize_map(Some(8))?;
        state.serialize_entry("name", &self.name)?;
        if let Some(fingerprint) = &self.fingerprint {
            state.serialize_entry("fingerprint", fingerprint)?;
        }
        state.serialize_entry("already_defined", &self.already_defined)?;
        if let Some(kind) = &self.kind {
            state.serialize_entry("kind", kind)?;
        }
        if let Some(size) = self.size {
            state.serialize_entry("size", &size)?;
        }
        if let Some(io_input) = self.io_input {
            state.serialize_entry("io_input", &io_input)?;
        }
        if let Some(io_output) = self.io_output {
            state.serialize_entry("io_output", &io_output)?;
        }
        state.serialize_entry("component_type", &self.component_type)?;
        if !self.parameters.is_empty() {
            state.serialize_entry("parameters", &self.parameters)?;
        }
        if let Some(isa) = &self.isa {
            state.serialize_entry("isa", isa)?;
        }
        state.end()
    }
}

#[derive(Clone)]
pub struct Resource {
    pub name: String,
    pub fingerprint: Option<String>,
    pub kind: Option<String>,
    pub already_defined: bool,
    pub slot: Option<u64>,
    pub size: Option<u64>,
    pub io_input: Option<bool>,
    pub io_output: Option<bool>,
    pub component_type: String,
    pub parameters: ParameterList,
    pub required_parameters: Vec<String>,
    pub isa: Option<InstructionSet>,
}

impl Resource {
    pub fn new(name: String, slot: Option<u64>, size: Option<u64>) -> Self {
        Resource {
            name,
            fingerprint: None,
            kind: None,
            already_defined: false,
            slot,
            size,
            io_input: None,
            io_output: None,
            component_type: "resource".to_string(),
            parameters: ParameterList::new(),
            required_parameters: Vec::new(),
            isa: None,
        }
    }

    pub fn is_valid(&self) -> Result<(), Error> {
        if self.name.is_empty() || self.kind.is_none() || self.size.is_none() || self.slot.is_none()
        {
            return Err(Error::ComponentWithoutNameOrKind);
        }
        if self.isa.is_none() {
            return Err(Error::ComponentWithoutISA);
        }
        if self.required_parameters.is_empty() {
            warn!("Resource {} has no required parameters", self.name);
        }

        Ok(())
    }

    pub fn from_json(json_str: &str) -> Result<Self, Error> {
        let json_value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        // Name
        if json_value.is_string() {
            let name = json_value.as_str().unwrap().to_string();
            return Ok(Resource::new(name, None, None));
        }
        let name: String;
        let name_option = json_value.get("resource_name");
        if name_option.is_none() {
            if json_value.get("name").is_some() {
                name = json_value["name"].as_str().unwrap().to_string();
            } else {
                name = "".to_string();
            }
        } else {
            name = name_option.unwrap().as_str().unwrap().to_string();
        }

        // Slot (optional)
        let slot = json_value.get("slot").map(|x| x.as_u64().unwrap());
        // Size (optional)
        let size = json_value.get("size").map(|x| x.as_u64().unwrap());
        let mut resource = Resource::new(name, slot, size);
        // Kind (optional)
        let resource_kind = json_value.get("kind");
        if let Some(resource_kind) = resource_kind {
            resource.kind = Some(resource_kind.as_str().unwrap().to_string());
        } else if resource.name.is_empty() {
            return Err(Error::ComponentWithoutNameOrKind);
        }

        // IO input (optional)
        let io_input = json_value.get("io_input");
        if let Some(io_input) = io_input {
            resource.io_input = Some(io_input.as_bool().unwrap());
        }
        // IO output (optional)
        let io_output = json_value.get("io_output");
        if let Some(io_output) = io_output {
            resource.io_output = Some(io_output.as_bool().unwrap());
        }

        // Check component type
        let component_type = json_value.get("component_type");
        if let Some(component_type) = component_type {
            if component_type.as_str().unwrap() != resource.component_type
                && !component_type.as_str().unwrap().is_empty()
            {
                if component_type.as_str().unwrap() == "controller" {
                    return Err(Error::ResourceDeclaredAsController);
                } else {
                    return Err(Error::UnknownComponentType);
                }
            }
        }
        // Parameters (optional)
        let json_resource_params = json_value.get("custom_properties");
        if let Some(json_resource_params) = json_resource_params {
            for param in json_resource_params.as_array().unwrap() {
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
                resource.add_parameter(name.to_string(), value);
            }
        }
        // Required parameters (optional)
        let required_parameters = json_value.get("required_parameters");
        if let Some(required_parameters) = required_parameters {
            for required_parameter in required_parameters.as_array().unwrap() {
                resource
                    .required_parameters
                    .push(required_parameter.as_str().unwrap().to_string());
            }
        }
        // ISA (optional)
        let isa = json_value.get("isa");
        if let Some(isa) = isa {
            let isa_result =
                InstructionSet::from_json(serde_json::from_str(isa.to_string().as_str()).unwrap());
            if let Ok(isa) = isa_result {
                resource.isa = Some(isa);
            } else {
                panic!("Error parsing ISA: {:?}", isa.to_string());
            }
        } else if let Some(kind) = &resource.kind {
            let lib_isa_json = get_isa_from_library(&kind.clone(), None).unwrap();
            let isa = InstructionSet::from_json(lib_isa_json);
            if isa.is_err() {
                panic!(
                    "Error with ISA for resource {} (kind: {}) cannot be found in library -> {}",
                    &resource.name,
                    kind,
                    isa.err().unwrap()
                );
            } else {
                resource.isa = Some(isa.unwrap());
            }
        }
        Ok(resource)
    }

    pub fn update_from_json(
        &mut self,
        json_str: &str,
        overwrite: bool,
    ) -> Result<Vec<(String, u64, u64)>, Error> {
        if let Ok(resource_from_pool) = Resource::from_json(json_str) {
            // Check slot
            if (self.slot.is_none() && resource_from_pool.slot.is_some()) || overwrite {
                self.slot = resource_from_pool.slot;
            }
            // Check size
            if (self.size.is_none() && resource_from_pool.size.is_some()) || overwrite {
                self.size = resource_from_pool.size;
            }

            // append parameters from resource_from_pool to resource required parameters
            for param_key in resource_from_pool.parameters.keys() {
                if !self.required_parameters.contains(param_key) {
                    self.required_parameters.push(param_key.clone());
                }
            }

            // append required parameters from resource_from_pool to resource
            if !resource_from_pool.required_parameters.is_empty() || overwrite {
                self.required_parameters
                    .extend(resource_from_pool.required_parameters);
            }
            // check if the resource has kind
            if (resource_from_pool.kind.is_some() && self.kind.is_none()) || overwrite {
                self.kind = resource_from_pool.kind;
            }
            // check io input
            if (resource_from_pool.io_input.is_some() && self.io_input.is_none()) || overwrite {
                self.io_input = resource_from_pool.io_input;
            }
            // check io output
            if (resource_from_pool.io_output.is_some() && self.io_output.is_none()) || overwrite {
                self.io_output = resource_from_pool.io_output;
            }

            // get isa from resource_from_pool if is not already defined
            if (resource_from_pool.isa.is_some() && self.isa.is_none()) || overwrite {
                self.isa = resource_from_pool.isa;
            }

            // append parameters from resource_from_pool to resource_parameters
            let overwritten_params =
                merge_parameters(&mut self.parameters, &resource_from_pool.parameters)?;
            return Ok(overwritten_params);
        }
        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Not implemented",
        )))
    }

    pub fn add_parameter(&mut self, name: String, value: u64) {
        self.parameters.insert(name, value);
    }
}

impl RTLComponent for Resource {
    fn generate_rtl(&self, output_folder: &Path) -> std::io::Result<()> {
        generate_rtl_for_component(
            self.kind.as_ref().unwrap(),
            self.name.as_str(),
            output_folder,
            &self,
        )
    }

    fn generate_bender(&self, output_folder: &Path) -> Result<(), Error> {
        let component_path = get_path_from_library(self.kind.as_ref().unwrap(), None).unwrap();
        let bender_filepath = Path::new(&component_path).join("Bender.yml");
        let component_with_hash: String =
            format!("{}_{}", self.name, self.clone().get_fingerprint());

        // Read the bender file
        let mut bender_file =
            serde_yml::from_str::<serde_yml::Value>(&fs::read_to_string(&bender_filepath).unwrap())
                .unwrap();
        bender_file["package"]["name"] = serde_yml::Value::String(component_with_hash);
        if !bender_filepath.exists() {
            panic!(
                "Bender file not found for component: {:?} (looking for {:?})",
                self.kind.as_ref().unwrap(),
                bender_filepath
            );
        }
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
        self.fingerprint = Some(generate_hash(vec![self.name.clone()], &self.parameters));
        self.fingerprint.clone().unwrap()
    }

    fn get_fingerprint(&mut self) -> String {
        if self.fingerprint.is_none() {
            self.generate_hash()
        } else {
            self.fingerprint.clone().unwrap()
        }
    }
}

impl Serialize for Resource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize the struct as a map with the following keys:
        // - resource_name
        // - kind
        // - slot
        // - size
        // - io_input
        // - io_output
        // - component_type
        // - custom_properties
        // - required_parameters

        let mut state = serializer.serialize_map(Some(8))?;
        state.serialize_entry("name", &self.name)?;
        if let Some(fingerprint) = &self.fingerprint {
            state.serialize_entry("fingerprint", fingerprint)?;
        }
        state.serialize_entry("already_defined", &self.already_defined)?;
        if let Some(kind) = &self.kind {
            state.serialize_entry("kind", kind)?;
        }
        if let Some(slot) = self.slot {
            state.serialize_entry("slot", &slot)?;
        }
        if let Some(size) = self.size {
            state.serialize_entry("size", &size)?;
        }
        if let Some(io_input) = self.io_input {
            state.serialize_entry("io_input", &io_input)?;
        }
        if let Some(io_output) = self.io_output {
            state.serialize_entry("io_output", &io_output)?;
        }
        state.serialize_entry("component_type", &self.component_type)?;
        if !self.parameters.is_empty() {
            state.serialize_entry("parameters", &self.parameters)?;
        }
        if let Some(isa) = &self.isa {
            state.serialize_entry("isa", isa)?;
        }
        state.end()
    }
}

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

    pub fn is_valid(&self) -> Result<(), Error> {
        if self.name.is_empty() || self.kind.is_none() {
            return Err(Error::ComponentWithoutNameOrKind);
        }
        if self.controller.is_none() {
            return Err(Error::CellWithoutController);
        }
        if self.isa.is_none() {
            return Err(Error::ComponentWithoutISA);
        }
        if self.resources.is_none() || self.resources.as_ref().unwrap().is_empty() {
            return Err(Error::CellWithoutResources);
        }
        Ok(())
    }

    pub fn from_json(json_str: &str) -> Result<Self, Error> {
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
            return Err(Error::ComponentWithoutNameOrKind);
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
                warn!("Error parsing controller: {:?}", controller.to_string());
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
                    warn!("Error parsing resource: {:?}", resource.to_string());
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
                panic!("Error parsing ISA: {:?}", isa.to_string());
            }
        } else if let Some(kind) = &cell.kind {
            let lib_isa_json = get_isa_from_library(&kind.clone(), None).unwrap();
            let isa = InstructionSet::from_json(lib_isa_json);
            if isa.is_err() {
                panic!(
                    "Error with ISA for cell {} (kind: {}) cannot be found in library -> {}",
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
    ) -> Result<Vec<(String, u64, u64)>, Error> {
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
        Err(Error::Io(std::io::Error::new(
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
    fn generate_rtl(&self, output_folder: &Path) -> std::io::Result<()> {
        generate_rtl_for_component(
            self.kind.as_ref().unwrap(),
            self.name.as_str(),
            output_folder,
            &self,
        )
    }

    fn generate_bender(&self, output_folder: &Path) -> Result<(), Error> {
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
        }
        if let Some(isa) = &self.isa {
            state.serialize_entry("isa", isa)?;
        }
        state.serialize_entry("fingerprint_table", &self.get_fingerprint_table())?;
        state.end()
    }
}

struct CellWithCoordinates {
    cell: Cell,
    coordinates: (u64, u64),
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

pub struct Fabric {
    pub cells: Vec<Vec<Cell>>,
    pub parameters: ParameterList,
}

impl Fabric {
    pub fn new() -> Self {
        Fabric {
            cells: Vec::new(),
            parameters: ParameterList::new(),
        }
    }

    pub fn add_parameters(&mut self, parameters: ParameterList) {
        // Check if ROWS and COLS are defined
        if parameters.contains_key("ROWS") && parameters.contains_key("COLS") {
            let rows = parameters.get("ROWS").unwrap();
            let cols = parameters.get("COLS").unwrap();
            self.cells =
                vec![vec![Cell::new("".to_string(), Vec::new()); *cols as usize]; *rows as usize];
        }
        self.parameters.extend(parameters);
    }

    pub fn add_cell(&mut self, cell: &Cell, row: u64, col: u64) {
        self.cells[row as usize][col as usize] = cell.clone();
    }

    pub fn get_parameter(&self, name: &str) -> Option<u64> {
        self.parameters.get(name).cloned()
    }

    fn get_fingerprint_table(&self) -> HashMap<String, String> {
        let mut fingerprint_table = HashMap::new();
        for row in self.cells.iter() {
            for cell in row.iter() {
                if let Some(fingerprint) = &cell.fingerprint {
                    if !fingerprint_table.contains_key(&cell.name) {
                        fingerprint_table.insert(cell.name.clone(), fingerprint.clone());
                    }
                    if !fingerprint_table.contains_key(&cell.controller.as_ref().unwrap().name) {
                        fingerprint_table.insert(
                            cell.controller.as_ref().unwrap().name.clone(),
                            cell.controller
                                .as_ref()
                                .unwrap()
                                .fingerprint
                                .clone()
                                .unwrap(),
                        );
                    }
                    if let Some(resources) = &cell.resources {
                        for resource in resources.iter() {
                            if !fingerprint_table.contains_key(&resource.name) {
                                fingerprint_table.insert(
                                    resource.name.clone(),
                                    resource.fingerprint.clone().unwrap(),
                                );
                            }
                        }
                    }
                }
            }
        }
        fingerprint_table
    }
}

impl RTLComponent for Fabric {
    fn generate_rtl(&self, output_folder: &Path) -> std::io::Result<()> {
        generate_rtl_for_component("fabric", "fabric", output_folder, &self)
    }

    fn generate_bender(&self, output_folder: &Path) -> Result<(), Error> {
        let component_path = get_path_from_library(&"fabric".to_string(), None).unwrap();
        let bender_filepath = Path::new(&component_path).join("Bender.yml");
        if !bender_filepath.exists() {
            panic!(
                "Bender file not found for fabric (looking for {:?})",
                bender_filepath
            );
        }

        // Read the bender file
        let mut bender_file =
            serde_yml::from_str::<serde_yml::Value>(&fs::read_to_string(&bender_filepath).unwrap())
                .unwrap();

        // Add controller and all resources to the dependencies
        let mut dependencies = serde_yml::Mapping::new();

        for row in self.cells.iter() {
            for cell in row.iter() {
                let cell_with_hash = format!("{}_{}", cell.name, cell.clone().get_fingerprint());
                let mut cell_path_map = serde_yml::Mapping::new();
                cell_path_map.insert(
                    serde_yml::Value::String("path".to_string()),
                    serde_yml::Value::String(format!("../cells/{}", cell_with_hash)),
                );
                dependencies.insert(
                    serde_yml::Value::String(cell_with_hash),
                    serde_yml::Value::Mapping(cell_path_map),
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
        let mut fingerprints = Vec::new();
        for row in self.cells.iter() {
            for cell in row.iter() {
                if let Some(fingerprint) = &cell.fingerprint {
                    fingerprints.push(fingerprint.clone());
                }
            }
        }
        generate_hash(fingerprints, &self.parameters)
    }

    fn get_fingerprint(&mut self) -> String {
        self.generate_hash()
    }
}

impl Serialize for Fabric {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize the struct as a map with the following keys:
        // - height
        // - width
        // - cells
        // - custom_properties
        let mut state = serializer.serialize_map(Some(4))?;
        state.serialize_entry(
            "__comment",
            "This file was automatically generated by Vesyla. DO NOT EDIT.",
        )?;
        let mut cells_with_coords = Vec::new();
        for (row_idx, row) in self.cells.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                let cell_with_coords = CellWithCoordinates {
                    cell: cell.clone(),
                    coordinates: (row_idx as u64, col_idx as u64),
                };
                cells_with_coords.push(cell_with_coords);
            }
        }
        state.serialize_entry("cells", &cells_with_coords)?;
        //state.serialize_entry("cells", &self.cells)?;
        if !self.parameters.is_empty() {
            state.serialize_entry("parameters", &self.parameters)?;
        }
        state.serialize_entry("fingerprint_table", &self.get_fingerprint_table())?;
        state.end()
    }
}
