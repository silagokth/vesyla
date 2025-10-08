use crate::{
    models::{
        drra::Fabric,
        pcu::Pcu,
        types::{DRRAError, ParameterList, RTLComponent},
    },
    utils::generate_hash,
};

use serde::ser::{Serialize, SerializeMap, Serializer};

use super::isa::InstructionSet;

pub struct Alimp {
    name: String,
    pub pcu: Pcu,
    pub drra: Option<Fabric>,
}

impl Alimp {
    pub fn new() -> Self {
        Self {
            name: "alimp".to_string(),
            pcu: Pcu::new(),
            drra: None,
        }
    }

    pub fn validate(&self) -> Result<(), DRRAError> {
        self.pcu.validate()?;
        if self.drra.is_some() {
            self.drra.as_ref().unwrap().validate()?;
        }

        Ok(())
    }

    pub fn get_isa(&self) -> Result<InstructionSet, DRRAError> {
        if let Some(drra) = self.drra.as_ref() {
            drra.get_isa()
        } else {
            Err(DRRAError::ParameterNotFound(
                "Alimp without DRRA fabrics do not have ISA".to_string(),
            ))
        }
    }
}

impl RTLComponent for Alimp {
    fn kind(&self) -> &str {
        "alimp"
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn generate_hash(&mut self) -> String {
        let mut fingerprints = Vec::new();
        let fabric = self.drra.as_mut().unwrap();

        let fingerprint = fabric.get_fingerprint();
        fingerprints.push(fingerprint);

        let params = ParameterList::new();

        generate_hash(fingerprints, &params)
    }

    fn get_fingerprint(&mut self) -> String {
        self.generate_hash()
    }
}

impl Serialize for Alimp {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(None)?;
        state.serialize_entry("pcu", &self.pcu)?;
        state.serialize_entry("drra", &self.drra)?;
        state.end()
    }
}
