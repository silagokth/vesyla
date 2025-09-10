use crate::models::{
    isa::ComponentInstructionSet,
    types::{DRRAError, ParameterList, RTLComponent},
};
use crate::utils::{generate_hash, get_path_from_library, merge_parameters};

use log::warn;
use std::fs;
use std::path::Path;

use serde::ser::{Serialize, SerializeMap, Serializer};

use super::isa::InstructionType;

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
    pub isa: Option<ComponentInstructionSet>,
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

    pub fn validate(&self) -> Result<(), DRRAError> {
        if self.name.is_empty() || self.kind.is_none() || self.size.is_none() || self.slot.is_none()
        {
            log::debug!(
                "Resource validation failed: name='{}', kind={:?}, size={:?}, slot={:?}",
                self.name,
                self.kind,
                self.size,
                self.slot
            );
            return Err(DRRAError::ComponentWithoutNameOrKind);
        }
        if self.isa.is_none() {
            return Err(DRRAError::ComponentWithoutISA);
        }
        if self.required_parameters.is_empty() {
            warn!("Resource {} has no required parameters", self.name);
        }

        Ok(())
    }

    pub fn overwrite(&mut self, other: &Resource) -> Result<(), DRRAError> {
        if self.name.is_empty() && !other.name.is_empty() {
            self.name = other.name.clone();
        }
        if other.fingerprint.is_some() {
            self.fingerprint = other.fingerprint.clone();
        }
        if other.kind.is_some() {
            self.kind = other.kind.clone();
        }
        if other.slot.is_some() {
            self.slot = other.slot;
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
        if other.component_type != self.component_type {
            return Err(DRRAError::UnknownComponentType);
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
            return Err(DRRAError::ComponentWithoutNameOrKind);
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
                    return Err(DRRAError::ResourceDeclaredAsController);
                } else {
                    return Err(DRRAError::UnknownComponentType);
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
            let comp_isa = ComponentInstructionSet::from_json(isa.clone())?;
            if let Some(ctype) = &comp_isa.component_type {
                if ctype != &InstructionType::ResourceInstruction {
                    return Err(DRRAError::ResourceDeclaredAsController);
                }
            }
            if comp_isa.component_kind.is_none() && resource.kind.is_some() {
                let mut isa_with_kind = comp_isa.clone();
                isa_with_kind.component_kind = resource.kind.clone();
                resource.isa = Some(isa_with_kind);
            } else if comp_isa.component_kind.is_some()
                && resource.kind.is_some()
                && comp_isa.component_kind.as_ref().unwrap() != resource.kind.as_ref().unwrap()
            {
                return Err(DRRAError::ISAKindMismatch);
            } else {
                resource.isa = Some(comp_isa);
            }
        }
        //else if let Some(kind) = &resource.kind {
        //    let lib_isa_json = get_isa_from_library(&kind.clone(), None).unwrap();
        //    let isa = ComponentInstructionSet::from_json(lib_isa_json);
        //    if isa.is_err() {
        //        panic!(
        //            "DRRAError with ISA for resource {} (kind: {}) cannot be found in library -> {}",
        //            &resource.name,
        //            kind,
        //            isa.err().unwrap()
        //        );
        //    } else {
        //        let mut isa_with_kind = isa.as_ref().unwrap().clone();
        //        isa_with_kind.component_kind = Some(kind.clone());
        //        resource.isa = Some(isa_with_kind);
        //    }
        //}
        Ok(resource)
    }

    pub fn update_from_json(
        &mut self,
        json_str: &str,
        overwrite: bool,
    ) -> Result<Vec<(String, u64, u64)>, DRRAError> {
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
        Err(DRRAError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Not implemented",
        )))
    }

    pub fn add_parameter(&mut self, name: String, value: u64) {
        self.parameters.insert(name, value);
    }
}

impl RTLComponent for Resource {
    fn kind(&self) -> &str {
        self.kind.as_ref().unwrap()
    }
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn generate_bender(&self, output_folder: &Path) -> Result<(), DRRAError> {
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
