use crate::utils::generate_rtl_for_component;

use std::{collections::BTreeMap, path::Path};

pub type ParameterList = BTreeMap<String, u64>;

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
    ParameterNotFound(String),
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
            DRRAError::ParameterNotFound(param) => {
                write!(f, "Parameter not found: {}", param)
            }
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
