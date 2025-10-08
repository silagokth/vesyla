use crate::models::cell::{Cell, CellWithCoordinates};
use crate::models::isa::InstructionSet;
use crate::models::types::{DRRAError, ParameterList, RTLComponent};
use crate::utils::generate_hash;

use std::{collections::HashMap, path::Path};

use hashlink::LinkedHashMap as LHashMap;
use serde::ser::{Serialize, SerializeMap, Serializer};
use yaml_rust2::Yaml;

#[derive(Clone)]
pub struct Fabric {
    pub cells: Vec<Vec<Cell>>,
    pub parameters: ParameterList,
}

impl Fabric {
    pub fn new() -> Self {
        Fabric {
            cells: Vec::new(),
            parameters: ParameterList::new(),
        }
    }

    pub fn add_parameters(&mut self, parameters: ParameterList) {
        // Check if ROWS and COLS are defined
        if parameters.contains_key("ROWS") && parameters.contains_key("COLS") {
            let rows = parameters.get("ROWS").unwrap();
            let cols = parameters.get("COLS").unwrap();
            self.cells =
                vec![vec![Cell::new("".to_string(), Vec::new()); *cols as usize]; *rows as usize];
        }
        self.parameters.extend(parameters);
    }

    pub fn add_cell(&mut self, cell: &Cell, row: u64, col: u64) {
        self.cells[row as usize][col as usize] = cell.clone();
    }

    pub fn get_parameter(&self, name: &str) -> Option<u64> {
        self.parameters.get(name).cloned()
    }

    pub fn get_isa(&self) -> Result<InstructionSet, DRRAError> {
        let mut instruction_set = InstructionSet::new();

        for row in self.cells.iter() {
            for cell in row.iter() {
                let mut cell = cell.clone();
                let cell_isa = cell.get_isa()?;
                instruction_set.add_format(cell_isa.format.clone())?;
                instruction_set.add_component_isas(cell_isa.component_instructions.clone())?;
            }
        }

        Ok(instruction_set)
    }

    pub fn generate_fingerprints(&mut self) -> Result<(), DRRAError> {
        for row in self.cells.iter_mut() {
            for cell in row.iter_mut() {
                cell.generate_fingerprints()?;
            }
        }

        self.generate_hash();

        Ok(())
    }

    pub fn validate(&self) -> Result<(), DRRAError> {
        for row in self.cells.iter() {
            for cell in row.iter() {
                cell.validate()?;
            }
        }

        Ok(())
    }

    fn get_fingerprint_table(&self) -> HashMap<String, String> {
        let mut fingerprint_table = HashMap::new();
        for row in self.cells.iter() {
            for cell in row.iter() {
                if let Some(fingerprint) = &cell.fingerprint {
                    if !fingerprint_table.contains_key(&cell.name) {
                        fingerprint_table.insert(cell.name.clone(), fingerprint.clone());
                    }
                    if !fingerprint_table.contains_key(&cell.controller.as_ref().unwrap().name) {
                        fingerprint_table.insert(
                            cell.controller.as_ref().unwrap().name.clone(),
                            cell.controller
                                .as_ref()
                                .unwrap()
                                .fingerprint
                                .clone()
                                .unwrap(),
                        );
                    }
                    if let Some(resources) = &cell.resources {
                        for resource in resources.iter() {
                            if !fingerprint_table.contains_key(&resource.name) {
                                fingerprint_table.insert(
                                    resource.name.clone(),
                                    resource.fingerprint.clone().unwrap(),
                                );
                            }
                        }
                    }
                }
            }
        }
        fingerprint_table
    }
}

impl RTLComponent for Fabric {
    fn generate_bender(&self, output_folder: &Path) -> Result<(), DRRAError> {
        let fingerprint = self.clone().get_fingerprint();
        let mut dependencies = LHashMap::new();
        for row in self.cells.iter() {
            for cell in row.iter() {
                let cell_with_hash = format!("{}_{}", cell.name, cell.clone().get_fingerprint());
                let mut cell_path_map = LHashMap::new();
                cell_path_map.insert(
                    Yaml::String("path".to_string()),
                    Yaml::String(format!("../cells/{}", cell_with_hash)),
                );
                dependencies.insert(Yaml::String(cell_with_hash), Yaml::Hash(cell_path_map));
            }
        }

        self.generate_bender_default(output_folder, Some(fingerprint), Some(dependencies))
    }

    fn generate_hash(&mut self) -> String {
        let mut fingerprints = Vec::new();
        for row in self.cells.iter() {
            for cell in row.iter() {
                if let Some(fingerprint) = &cell.fingerprint {
                    fingerprints.push(fingerprint.clone());
                }
            }
        }
        generate_hash(fingerprints, &self.parameters)
    }

    fn get_fingerprint(&mut self) -> String {
        self.generate_hash()
    }

    fn kind(&self) -> &str {
        "fabric"
    }

    fn name(&self) -> &str {
        "fabric"
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
        state.serialize_entry(
            "__comment",
            "This file was automatically generated by Vesyla. DO NOT EDIT.",
        )?;
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
        state.serialize_entry("fingerprint_table", &self.get_fingerprint_table())?;
        state.end()
    }
}
