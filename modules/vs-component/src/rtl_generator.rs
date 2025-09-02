use crate::{
    models::{
        cell::Cell, controller::Controller, drra::Fabric, resource::Resource, types::RTLComponent,
    },
    resolver::ResolvedAlimp,
};

use std::{collections::HashMap, fs, io::Error, path::Path, path::PathBuf};

pub struct RTLGenerator {
    rtl_output_dir: PathBuf,
    implemented_cells: HashMap<String, Cell>,
    implemented_resources: HashMap<String, Resource>,
    implemented_controllers: HashMap<String, Controller>,
}

impl RTLGenerator {
    pub fn new(rtl_output_dir: &Path) -> Self {
        Self {
            rtl_output_dir: rtl_output_dir.to_path_buf(),
            implemented_cells: HashMap::new(),
            implemented_resources: HashMap::new(),
            implemented_controllers: HashMap::new(),
        }
    }

    pub fn generate(&mut self, resolved_alimp: &mut ResolvedAlimp) -> Result<(), Error> {
        for cell in &mut resolved_alimp.resolved_cells {
            self.process_resolved_cell(cell)?;
        }

        self.generate_fabric_rtl(resolved_alimp.alimp.drra.as_mut().unwrap())?;

        Ok(())
    }

    fn process_resolved_cell(&mut self, cell: &mut Cell) -> Result<(), Error> {
        if let Some(controller) = &mut cell.controller {
            self.process_resolved_controller(controller)?;
        } else {
            return Err(Error::new(
                std::io::ErrorKind::Other,
                format!("Cell {} does not have a controller assigned", cell.name),
            ));
        }

        if let Some(resources) = &mut cell.resources {
            for resource in resources {
                self.process_resolved_resource(resource)?;
            }
        } else {
            return Err(Error::new(
                std::io::ErrorKind::Other,
                format!("Cell {} does not have resources assigned", cell.name),
            ));
        }

        let cell_hash = cell.get_fingerprint();
        if !self.implemented_cells.contains_key(&cell_hash) {
            self.implemented_cells
                .insert(cell_hash.clone(), cell.clone());

            let cell_with_hash = format!("{}_{}", cell.name, cell_hash);
            let bender_output_folder = self.rtl_output_dir.join("cells").join(&cell_with_hash);
            let rtl_output_folder = bender_output_folder.join("rtl");

            // Generate the cell RTL
            cell.generate_rtl(&rtl_output_folder)?;
            cell.generate_bender(&bender_output_folder).map_err(|e| {
                Error::new(
                    std::io::ErrorKind::Other,
                    format!("While generating Bender file for cell: {}", e),
                )
            })?;
        } else {
            cell.already_defined = true;
        }

        Ok(())
    }

    fn process_resolved_controller(&mut self, controller: &mut Controller) -> Result<(), Error> {
        let controller_hash = controller.get_fingerprint();

        if !self.implemented_controllers.contains_key(&controller_hash) {
            self.implemented_controllers
                .insert(controller_hash.clone(), controller.clone());

            let controller_with_hash = format!("{}_{}", controller.name, controller_hash);
            let bender_output_folder = self
                .rtl_output_dir
                .join("controllers")
                .join(&controller_with_hash);
            let rtl_output_folder = bender_output_folder.join("rtl");

            controller.generate_rtl(&rtl_output_folder)?;
            controller
                .generate_bender(&bender_output_folder)
                .map_err(|e| {
                    Error::new(
                        std::io::ErrorKind::Other,
                        format!("While generating Bender file for controller: {}", e),
                    )
                })?;
        } else {
            controller.already_defined = true;
        }

        Ok(())
    }

    fn process_resolved_resource(&mut self, resource: &mut Resource) -> Result<(), Error> {
        let resource_hash = resource.get_fingerprint();

        if !self.implemented_resources.contains_key(&resource_hash) {
            self.implemented_resources
                .insert(resource_hash.clone(), resource.clone());

            let resource_with_hash = format!("{}_{}", resource.name, resource_hash);
            let bender_output_folder = self
                .rtl_output_dir
                .join("resources")
                .join(&resource_with_hash);
            let rtl_output_folder = bender_output_folder.join("rtl");

            resource.generate_rtl(&rtl_output_folder)?;
            resource
                .generate_bender(&bender_output_folder)
                .map_err(|e| {
                    Error::new(
                        std::io::ErrorKind::Other,
                        format!("While generating Bender file for resource: {}", e),
                    )
                })?;
        } else {
            resource.already_defined = true;
        }

        Ok(())
    }

    fn generate_fabric_rtl(&self, fabric: &Fabric) -> Result<(), Error> {
        // Output the fabric RTL
        let output_folder = &self.rtl_output_dir.join("fabric");
        // Remove files in the output directory
        if output_folder.exists() {
            fs::remove_dir_all(output_folder)?;
        }
        let rtl_output_folder = output_folder.join("rtl");

        fabric.generate_rtl(&rtl_output_folder)?;
        fabric.generate_bender(output_folder).map_err(|e| {
            Error::new(
                std::io::ErrorKind::Other,
                format!("While generating Bender file for fabric: {}", e),
            )
        })?;

        Ok(())
    }
}
