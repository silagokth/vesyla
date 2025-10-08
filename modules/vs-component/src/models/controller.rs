use crate::models::{
    isa::{ComponentInstructionSet, InstructionType},
    types::{DRRAError, ParameterList, RTLComponent},
};
use crate::utils::{generate_hash, merge_parameters};

use log::warn;
use std::path::Path;

use serde::ser::{Serialize, SerializeMap, Serializer};

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
    pub isa: Option<ComponentInstructionSet>,
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

    pub fn validate(&self) -> Result<(), DRRAError> {
        if self.name.is_empty() || self.kind.is_none() || self.size.is_none() {
            log::debug!(
                "Controller validation failed: name='{}', kind={:?}, size={:?}",
                self.name,
                self.kind,
                self.size
            );
            return Err(DRRAError::ComponentWithoutNameOrKind);
        }
        if self.isa.is_none() {
            return Err(DRRAError::ComponentWithoutISA);
        }
        if self.required_parameters.is_empty() {
            warn!("Controller {} has no required parameters", self.name);
        }
        Ok(())
    }

    pub fn overwrite(&mut self, other: &Controller) -> Result<(), DRRAError> {
        if self.name.is_empty() && !other.name.is_empty() {
            self.name = other.name.clone();
        }
        if other.fingerprint.is_some() {
            self.fingerprint = other.fingerprint.clone();
        }
        if other.kind.is_some() {
            self.kind = other.kind.clone();
        }
        if other.size.is_some() {
            self.size = other.size;
        }
        if other.io_input.is_some() {
            self.io_input = other.io_input;
        }
        if other.io_output.is_some() {
            self.io_output = other.io_output;
        }
        if other.component_type != self.component_type {
            return Err(DRRAError::UnknownComponentType);
        }
        if other.isa.is_some() {
            if self.isa.is_none() {
                self.isa = other.isa.clone();
                self.isa.as_mut().unwrap().component_kind = self.kind.clone();
            } else {
                self.isa.as_mut().unwrap().component_kind = self.kind.clone();
                self.isa
                    .as_mut()
                    .unwrap()
                    .overwrite(other.isa.clone().unwrap())?;
            }
        }
        for (key, value) in &other.parameters {
            self.parameters.insert(key.clone(), *value);
        }
        for param in &other.required_parameters {
            if !self.required_parameters.contains(param) {
                self.required_parameters.push(param.clone());
            }
        }

        Ok(())
    }

    pub fn from_json(json_str: &str) -> Result<Self, DRRAError> {
        let json_value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        // Name
        if json_value.is_string() {
            let name = json_value.as_str().unwrap().to_string();
            return Ok(Controller::new(name, None));
        }
        let name: String;
        let name_option = json_value.get("controller_name"); //].as_str().unwrap().to_string();
        if let Some(name_option) = name_option {
            name = name_option.as_str().unwrap().to_string();
        } else if json_value.get("name").is_some() {
            name = json_value["name"].as_str().unwrap().to_string();
        } else {
            name = "".to_string();
        }

        // Size (optional)
        let size = json_value.get("size").map(|x| x.as_u64().unwrap());
        let mut controller = Controller::new(name, size);

        // Kind (optional)
        let controller_kind = json_value.get("kind");
        if let Some(controller_kind) = controller_kind {
            controller.kind = Some(controller_kind.as_str().unwrap().to_string());
        } else if controller.name.is_empty() {
            return Err(DRRAError::ComponentWithoutNameOrKind);
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
                    return Err(DRRAError::ControllerDeclaredAsResource);
                } else {
                    return Err(DRRAError::UnknownComponentType);
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
            let comp_isa = ComponentInstructionSet::from_json(isa.clone())?;
            if let Some(ctype) = &comp_isa.component_type {
                if ctype != &InstructionType::ControlInstruction {
                    return Err(DRRAError::ControllerDeclaredAsResource);
                }
            }
            if comp_isa.component_kind.is_none() && controller.kind.is_some() {
                let mut isa_with_kind = comp_isa.clone();
                isa_with_kind.component_kind = controller.kind.clone();
                controller.isa = Some(isa_with_kind);
            } else if comp_isa.component_kind.is_some()
                && controller.kind.is_some()
                && comp_isa.component_kind.as_ref().unwrap() != controller.kind.as_ref().unwrap()
            {
                return Err(DRRAError::ISAKindMismatch);
            } else {
                controller.isa = Some(comp_isa);
            }
        }
        //else if let Some(kind) = &controller.kind {
        //    let lib_isa_json = get_isa_from_library(&kind.clone(), None).unwrap();
        //    let isa = ComponentInstructionSet::from_json(lib_isa_json);
        //    if isa.is_err() {
        //        panic!(
        //            "DRRAError with ISA for controller {} (kind: {}) cannot be found in library -> {}",
        //            &controller.name,
        //            kind,
        //            isa.err().unwrap()
        //        );
        //    } else {
        //        let mut isa_with_kind = isa.as_ref().unwrap().clone();
        //        isa_with_kind.component_kind = Some(kind.clone());
        //        controller.isa = Some(isa_with_kind);
        //    }
        //}
        Ok(controller)
    }

    pub fn update_from_json(
        &mut self,
        json_str: &str,
        overwrite: bool,
    ) -> Result<Vec<(String, u64, u64)>, DRRAError> {
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
        Err(DRRAError::Io(std::io::Error::other("Not implemented")))
    }

    pub fn add_parameter(&mut self, name: String, value: u64) {
        self.parameters.insert(name, value);
    }
}

impl RTLComponent for Controller {
    fn kind(&self) -> &str {
        self.kind.as_ref().unwrap()
    }
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn generate_bender(&self, output_folder: &Path) -> Result<(), DRRAError> {
        let fingerprint = self.clone().get_fingerprint();
        self.generate_bender_default(output_folder, Some(fingerprint), None)
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
        let mut state = serializer.serialize_map(Some(10))?;
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
