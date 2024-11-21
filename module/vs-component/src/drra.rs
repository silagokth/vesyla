use crate::utils::{get_library_path, get_rtl_template_from_library};
use log::warn;
use serde::ser::{Serialize, SerializeMap, Serializer};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::write,
    fs,
    path::{Path, PathBuf},
};

pub type ParameterList = BTreeMap<String, u64>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    ResourceDeclaredAsController,
    ControllerDeclaredAsResource,
    ComponentWithoutNameOrKind,
    UnknownComponentType,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ResourceDeclaredAsController => write!(f, "Resource declared as controller"),
            Error::ControllerDeclaredAsResource => write!(f, "Controller declared as resource"),
            Error::ComponentWithoutNameOrKind => write!(f, "Component without name or kind"),
            Error::UnknownComponentType => write!(f, "Unknown component type"),
            Error::Io(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl std::convert::From<std::io::Error> for Error {
    fn from(_error: std::io::Error) -> Self {
        Error::Io(_error)
    }
}

pub trait RTLComponent {
    fn generate_rtl(&self, output_file: &Path) -> Result<(), Error>;
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
        }
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
        Ok(controller)
    }

    pub fn add_parameter(&mut self, name: String, value: u64) {
        self.parameters.insert(name, value);
    }
}

impl RTLComponent for Controller {
    fn generate_rtl(&self, output_file: &Path) -> Result<(), Error> {
        // Get the RTL template for the controller
        if let Ok(rtl_template) = get_rtl_template_from_library(self.kind.as_ref().unwrap()) {
            // Create output file
            fs::File::create(output_file).expect("Failed to create file");
            let mj_env = minijinja::Environment::new();
            if let Ok(output_str) = mj_env.render_str(&rtl_template, self) {
                fs::write(output_file, output_str)?;
            } else {
                panic!("Failed to render template for controller {}", self.name);
            }
        } else {
            panic!("Failed to get RTL template for controller {}", self.name);
        }
        Ok(())
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
        }
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
            let json_params_list = json_resource_params.as_array().unwrap();
            if json_params_list.is_empty() {
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
        Ok(resource)
    }
    pub fn add_parameter(&mut self, name: String, value: u64) {
        self.parameters.insert(name, value);
    }
}

impl RTLComponent for Resource {
    fn generate_rtl(&self, output_file: &Path) -> Result<(), Error> {
        // Get the RTL template for the controller
        if let Ok(rtl_template) = get_rtl_template_from_library(self.kind.as_ref().unwrap()) {
            // Create output file
            fs::File::create(output_file).expect("Failed to create file");
            let mj_env = minijinja::Environment::new();
            if let Ok(output_str) = mj_env.render_str(&rtl_template, self) {
                fs::write(output_file, output_str)?;
            } else {
                panic!("Failed to render template for controller {}", self.name);
            }
        } else {
            panic!("Failed to get RTL template for controller {}", self.name);
        }
        Ok(())
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
        }
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
        Ok(cell)
    }

    pub fn add_parameter(&mut self, name: String, value: u64) {
        self.parameters.insert(name, value);
    }

    //pub fn get_resources_names(&self) -> Vec<String> {
    //    let mut resources_names = Vec::new();
    //    if let Some(resources) = &self.resources {
    //        for resource in resources {
    //            resources_names.push(resource.name.clone());
    //        }
    //    }
    //    resources_names
    //}
}

impl RTLComponent for Cell {
    fn generate_rtl(&self, output_file: &Path) -> Result<(), Error> {
        // Get the RTL template for the controller
        if let Ok(rtl_template) = get_rtl_template_from_library(self.kind.as_ref().unwrap()) {
            // Create output file
            fs::File::create(output_file).expect("Failed to create file");
            let mj_env = minijinja::Environment::new();
            if let Ok(output_str) = mj_env.render_str(&rtl_template, self) {
                fs::write(output_file, output_str)?;
            } else {
                panic!("Failed to render template for controller {}", self.name);
            }
        } else {
            panic!("Failed to get RTL template for controller {}", self.name);
        }
        Ok(())
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
    pub height: u64,
    pub width: u64,
    pub cells: Vec<Vec<Cell>>,
    pub parameters: ParameterList,
}

impl Fabric {
    pub fn new(height: u64, width: u64) -> Self {
        let mut cells = Vec::new();
        for _ in 0..height {
            let mut row = Vec::new();
            for _ in 0..width {
                row.push(Cell::new("".to_string(), vec![]));
            }
            cells.push(row);
        }
        Fabric {
            height,
            width,
            cells,
            parameters: ParameterList::new(),
        }
    }

    pub fn add_cell(&mut self, cell: &Cell, row: u64, col: u64) {
        self.cells[row as usize][col as usize] = cell.clone();
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
        state.serialize_entry("height", &self.height)?;
        state.serialize_entry("width", &self.width)?;
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
        state.end()
    }
}
