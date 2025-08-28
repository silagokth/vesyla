use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::models::drra::RTLComponent;

pub struct Pcu {
    pub has_io_connection: bool,
    pub has_vpi: bool,
}

impl Pcu {
    pub fn new() -> Self {
        Self {
            has_io_connection: false,
            has_vpi: false,
        }
    }
}

impl RTLComponent for Pcu {
    fn generate_rtl(&self, _output_folder: &std::path::Path) -> std::io::Result<()> {
        todo!()
    }

    fn generate_bender(
        &self,
        output_folder: &std::path::Path,
    ) -> Result<(), crate::models::drra::DRRAError> {
        todo!()
    }

    fn generate_hash(&mut self) -> String {
        todo!()
    }

    fn get_fingerprint(&mut self) -> String {
        todo!()
    }
}

impl Serialize for Pcu {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(None)?;
        state.serialize_entry("has_io_connection", &self.has_io_connection)?;
        state.serialize_entry("has_vpi", &self.has_vpi)?;
        state.end()
    }
}
