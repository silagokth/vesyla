use log::warn;
use std::{collections::BTreeMap, fmt::write};

pub type ParameterList = BTreeMap<String, u64>;

#[derive(Clone)]
pub struct Controller {
    pub name: String,
    pub kind: Option<String>,
    pub size: Option<u64>,
    pub parameters: ParameterList,
    pub required_parameters: Vec<String>,
}

impl Controller {
    pub fn new(name: String, size: Option<u64>) -> Self {
        Controller {
            name,
            kind: None,
            size,
            parameters: ParameterList::new(),
            required_parameters: Vec::new(),
        }
    }

    pub fn from_json(json_str: &str) -> Result<Self, std::io::Error> {
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
                panic!("Controller name not found");
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

#[derive(Clone)]
pub struct Resource {
    pub name: String,
    pub kind: Option<String>,
    pub slot: Option<u64>,
    pub size: Option<u64>,
    pub parameters: ParameterList,
    pub required_parameters: Vec<String>,
}

impl Resource {
    pub fn new(name: String, slot: Option<u64>, size: Option<u64>) -> Self {
        Resource {
            name,
            kind: None,
            slot,
            size,
            parameters: ParameterList::new(),
            required_parameters: Vec::new(),
        }
    }
    pub fn from_json(json_str: &str) -> Result<Self, std::io::Error> {
        let json_value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        // Name
        if json_value.is_string() {
            let name = json_value.as_str().unwrap().to_string();
            return Ok(Resource::new(name, None, None));
        }
        let name: String;
        let name_option = json_value.get("resource_name"); //].as_str().unwrap().to_string();
        if name_option.is_none() {
            if json_value.get("name").is_some() {
                name = json_value["name"].as_str().unwrap().to_string();
            } else {
                panic!("Resource name not found");
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

#[derive(Clone)]
pub struct Cell {
    pub name: String,
    pub coordinates_list: Vec<(u64, u64)>,
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
            coordinates_list,
            kind: None,
            controller: None,
            resources: None,
            parameters: ParameterList::new(),
            required_parameters: Vec::new(),
        }
    }

    pub fn from_json(json_str: &str) -> Result<Self, std::io::Error> {
        let json_value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        // Name
        let name: String;
        let name_option = json_value.get("cell_name"); //].as_str().unwrap().to_string();
        if name_option.is_none() {
            if json_value.get("name").is_some() {
                name = json_value["name"].as_str().unwrap().to_string();
            } else {
                panic!("Cell name not found");
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

        // Cell kind (optional)
        let cell_kind = json_value.get("kind");
        if let Some(cell_kind) = cell_kind {
            cell.kind = Some(cell_kind.as_str().unwrap().to_string());
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

//impl std::fmt::Debug for Cell {
//    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//        // Print following this format:
//        //{
//        //"coordinates": [
//        //  {
//        //    "row": 0,
//        //    "col": 0
//        //  }
//        //],
//        //"cell_name": "drra_cell_input"
//        //}
//        write!(f, "{{\n\"coordinates\": [\n{{\n\"row\": {},\n\"col\": {}\n}}\n],\n\"cell_name\": \"{}\"\n}}", self.coordinates.0, self.coordinates.1, self.name)
//    }
//}
