use crate::models::types::{DRRAError, ParameterList, RTLComponent};
use crate::utils::generate_hash;

use serde::ser::{Serialize, SerializeMap, Serializer};

pub struct Pcu {
    name: String,
    pub has_ssm: bool,
    pub has_vpi: bool,
}

impl Pcu {
    pub fn new() -> Self {
        Self {
            name: "pcu".to_string(),
            has_ssm: false,
            has_vpi: false,
        }
    }

    pub fn validate(&self) -> Result<(), DRRAError> {
        if self.has_vpi && !self.has_ssm {
            return Err(DRRAError::ParameterNotFound(
                "PCU cannot have VPI without SSM".to_string(),
            ));
        }

        Ok(())
    }
}

impl RTLComponent for Pcu {
    fn kind(&self) -> &str {
        "pcu"
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn generate_hash(&mut self) -> String {
        let fingerprints: Vec<String> = Vec::new();
        let mut parameter_list = ParameterList::new();
        parameter_list.insert("has_ssm".to_string(), self.has_ssm as u64);
        parameter_list.insert("has_vpi".to_string(), self.has_vpi as u64);
        generate_hash(fingerprints, &parameter_list)
    }

    fn get_fingerprint(&mut self) -> String {
        self.generate_hash()
    }
}

impl Serialize for Pcu {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(None)?;
        state.serialize_entry("has_io_connection", &self.has_ssm)?;
        state.serialize_entry("has_vpi", &self.has_vpi)?;
        state.end()
    }
}
