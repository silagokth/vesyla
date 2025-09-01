use crate::utils::generate_rtl_for_component;

use std::io::Error;
use std::{collections::BTreeMap, path::Path};

pub type ParameterList = BTreeMap<String, u64>;

pub trait ParameterListExt {
    fn add_parameter(&mut self, key: String, value: u64) -> Result<bool, Error>;
    fn add_parameters(&mut self, parameters_to_add: &ParameterList)
        -> Result<ParameterList, Error>;
    fn remove_parameters(&mut self, parameters: &ParameterList) -> Result<(), Error>;
}

impl ParameterListExt for ParameterList {
    fn add_parameter(&mut self, key: String, value: u64) -> Result<bool, Error> {
        if self.contains_key(key.as_str()) {
            if self.get(key.as_str()).unwrap() != &value {
                return Err(Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "Duplicate parameter with different value: {} ({} vs. {})",
                        key,
                        self.get(key.as_str()).unwrap(),
                        value
                    ),
                ));
            } else {
                return Ok(false);
            }
        }

        self.insert(key, value);
        Ok(true)
    }

    fn add_parameters(
        &mut self,
        parameters_to_add: &ParameterList,
    ) -> Result<ParameterList, Error> {
        let mut added_params = ParameterList::new();
        for (param_name, param_value) in parameters_to_add.iter() {
            match self.add_parameter(param_name.to_string(), *param_value) {
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

    fn remove_parameters(&mut self, parameters: &ParameterList) -> Result<(), Error> {
        for (param_name, _) in parameters.iter() {
            self.remove(param_name);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum DRRAError {
    Io(std::io::Error),
    ResourceDeclaredAsController,
    ControllerDeclaredAsResource,
    ComponentWithoutNameOrKind,
    ComponentWithoutISA,
    UnknownComponentType,
    CellWithoutController,
    CellWithoutResources,
}

impl std::fmt::Display for DRRAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DRRAError::ResourceDeclaredAsController => write!(f, "Resource declared as controller"),
            DRRAError::ControllerDeclaredAsResource => write!(f, "Controller declared as resource"),
            DRRAError::ComponentWithoutNameOrKind => write!(f, "Component without name or kind"),
            DRRAError::ComponentWithoutISA => write!(f, "Component without ISA"),
            DRRAError::UnknownComponentType => write!(f, "Unknown component type"),
            DRRAError::Io(err) => write!(f, "IO error: {}", err),
            DRRAError::CellWithoutController => write!(f, "Cell without controller"),
            DRRAError::CellWithoutResources => write!(f, "Cell without resources"),
        }
    }
}

impl std::convert::From<std::io::Error> for DRRAError {
    fn from(_error: std::io::Error) -> Self {
        DRRAError::Io(_error)
    }
}

pub trait RTLComponent {
    fn kind(&self) -> &str;
    fn name(&self) -> &str;
    fn generate_rtl(&self, output_folder: &Path) -> std::io::Result<()>
    where
        Self: serde::Serialize,
    {
        generate_rtl_for_component(self.kind(), self.name(), output_folder, &self)
    }
    fn generate_bender(&self, output_folder: &Path) -> Result<(), DRRAError>;
    fn generate_hash(&mut self) -> String;
    fn get_fingerprint(&mut self) -> String;
}
