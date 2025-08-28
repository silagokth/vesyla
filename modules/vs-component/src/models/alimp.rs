use crate::models::{
    drra::{Fabric, RTLComponent},
    pcu::Pcu,
};

use serde::ser::{Serialize, SerializeMap, Serializer};

pub struct ALImp {
    pub pcu: Pcu,
    pub drra: Option<Fabric>,
}

impl ALImp {
    pub fn new() -> Self {
        Self {
            pcu: Pcu::new(),
            drra: None,
        }
    }
}

impl RTLComponent for ALImp {
    fn generate_rtl(&self, output_folder: &std::path::Path) -> std::io::Result<()> {
        todo!()
    }

    fn generate_bender(
        &self,
        output_folder: &std::path::Path,
    ) -> Result<(), super::drra::DRRAError> {
        todo!()
    }

    fn generate_hash(&mut self) -> String {
        todo!()
    }

    fn get_fingerprint(&mut self) -> String {
        todo!()
    }
}

impl Serialize for ALImp {
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
