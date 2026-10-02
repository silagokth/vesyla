//! Load an elaborated fabric back from the arch.json `elaborate` writes, or from
//! a copy of it whose parameters were changed -- the sized arch.json the
//! compiler writes for one program.
//!
//! Every cell, controller and resource in the file names its `kind`, and that
//! kind's library entry is where whatever the file leaves out comes from: the
//! ISA, and the default of any parameter the file does not give. A parameter
//! missing from a component falls back the way the resolver resolves it --
//! the component's own library default, then its cell's parameters, then the
//! fabric's. Fingerprints, ISAs and fingerprint tables stored in the file are
//! ignored and recomputed, since a changed parameter makes them stale.

use crate::models::{
    cell::Cell,
    controller::Controller,
    drra::Fabric,
    isa::ComponentInstructionSet,
    resource::Resource,
    types::{DRRAError, ParameterList, RTLComponent},
};
use crate::utils::{get_arch_from_library, get_isa_from_library, get_parameters};

use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Error,
    path::Path,
};

pub struct FabricLoader {
    library_cache: HashMap<String, serde_json::Value>,
    isa_cache: HashMap<String, ComponentInstructionSet>,
}

impl FabricLoader {
    pub fn new() -> Self {
        Self {
            library_cache: HashMap::new(),
            isa_cache: HashMap::new(),
        }
    }

    pub fn load(&mut self, arch_json_path: &Path) -> Result<Fabric, DRRAError> {
        let input: serde_json::Value = serde_json::from_reader(fs::File::open(arch_json_path)?)
            .map_err(|e| DRRAError::Io(Error::new(std::io::ErrorKind::InvalidInput, e)))?;

        let mut fabric = Fabric::new();
        fabric.add_parameters(get_parameters(&input, None));
        if fabric.cells.is_empty() {
            return Err(invalid(format!(
                "{} has no ROWS and COLS parameters -- it must be an elaborated arch.json",
                arch_json_path.display()
            )));
        }

        let cells = input
            .get("cells")
            .and_then(|cells| cells.as_array())
            .ok_or_else(|| {
                invalid(format!(
                    "{} has no \"cells\" list -- it must be an elaborated arch.json",
                    arch_json_path.display()
                ))
            })?;

        for entry in cells {
            let row = entry["coordinates"]["row"].as_u64();
            let col = entry["coordinates"]["col"].as_u64();
            let (row, col) = match (row, col) {
                (Some(row), Some(col)) => (row, col),
                _ => {
                    return Err(invalid(
                        "a cell entry has no (row, col) coordinates".to_string(),
                    ));
                }
            };
            if row as usize >= fabric.cells.len() || col as usize >= fabric.cells[0].len() {
                return Err(invalid(format!(
                    "cell ({}, {}) lies outside the {}x{} fabric",
                    row,
                    col,
                    fabric.cells.len(),
                    fabric.cells[0].len()
                )));
            }
            let cell = self.load_cell(&entry["cell"], &fabric.parameters)?;
            fabric.add_cell(&cell, row, col);
        }

        fabric.generate_fingerprints()?;
        make_names_unique(&mut fabric)?;
        fabric.validate()?;

        Ok(fabric)
    }

    fn load_cell(
        &mut self,
        cell_json: &serde_json::Value,
        fabric_parameters: &ParameterList,
    ) -> Result<Cell, DRRAError> {
        let name = string_field(cell_json, "name", "cell")?;
        let kind = string_field(cell_json, "kind", &name)?;
        let from_lib = Cell::from_json(&self.library_arch(&kind)?.to_string())?;

        let mut cell = Cell::new(name, Vec::new());
        cell.kind = Some(kind);
        cell.io_input = bool_field(cell_json, "io_input").or(from_lib.io_input);
        cell.io_output = bool_field(cell_json, "io_output").or(from_lib.io_output);

        let own = get_parameters(cell_json, None);
        cell.required_parameters =
            required_parameters(&from_lib.required_parameters, &from_lib.parameters, &own);
        cell.parameters = fill_parameters(
            &cell.name,
            &cell.required_parameters,
            &own,
            &from_lib.parameters,
            &[fabric_parameters],
        )?;

        let controller_json = cell_json
            .get("controller")
            .ok_or(DRRAError::CellWithoutController)?;
        cell.controller =
            Some(self.load_controller(controller_json, &cell.parameters, fabric_parameters)?);

        let resources_json = cell_json
            .get("resources_list")
            .and_then(|resources| resources.as_array())
            .ok_or(DRRAError::CellWithoutResources)?;
        let mut resources = Vec::new();
        for resource_json in resources_json {
            resources.push(self.load_resource(
                resource_json,
                &cell.parameters,
                fabric_parameters,
            )?);
        }
        cell.resources = Some(resources);

        cell.get_isa()?;

        Ok(cell)
    }

    fn load_controller(
        &mut self,
        controller_json: &serde_json::Value,
        cell_parameters: &ParameterList,
        fabric_parameters: &ParameterList,
    ) -> Result<Controller, DRRAError> {
        let name = string_field(controller_json, "name", "controller")?;
        let kind = string_field(controller_json, "kind", &name)?;
        let from_lib = Controller::from_json(&self.library_arch(&kind)?.to_string())?;

        let size = controller_json
            .get("size")
            .and_then(|size| size.as_u64())
            .or(from_lib.size);
        let mut controller = Controller::new(name, size);
        controller.kind = Some(kind.clone());
        controller.io_input = bool_field(controller_json, "io_input").or(from_lib.io_input);
        controller.io_output = bool_field(controller_json, "io_output").or(from_lib.io_output);

        let own = get_parameters(controller_json, None);
        controller.required_parameters =
            required_parameters(&from_lib.required_parameters, &from_lib.parameters, &own);
        controller.parameters = fill_parameters(
            &controller.name,
            &controller.required_parameters,
            &own,
            &from_lib.parameters,
            &[cell_parameters, fabric_parameters],
        )?;
        controller.isa = Some(self.library_isa(&kind)?);

        controller.validate()?;

        Ok(controller)
    }

    fn load_resource(
        &mut self,
        resource_json: &serde_json::Value,
        cell_parameters: &ParameterList,
        fabric_parameters: &ParameterList,
    ) -> Result<Resource, DRRAError> {
        let name = string_field(resource_json, "name", "resource")?;
        let kind = string_field(resource_json, "kind", &name)?;
        let from_lib = Resource::from_json(&self.library_arch(&kind)?.to_string())?;

        let slot = resource_json
            .get("slot")
            .and_then(|slot| slot.as_u64())
            .ok_or_else(|| invalid(format!("resource '{}' has no slot", name)))?;
        let size = resource_json
            .get("size")
            .and_then(|size| size.as_u64())
            .or(from_lib.size);
        let mut resource = Resource::new(name, Some(slot), size);
        resource.kind = Some(kind.clone());
        resource.io_input = bool_field(resource_json, "io_input").or(from_lib.io_input);
        resource.io_output = bool_field(resource_json, "io_output").or(from_lib.io_output);

        let own = get_parameters(resource_json, None);
        resource.required_parameters =
            required_parameters(&from_lib.required_parameters, &from_lib.parameters, &own);
        resource.parameters = fill_parameters(
            &resource.name,
            &resource.required_parameters,
            &own,
            &from_lib.parameters,
            &[cell_parameters, fabric_parameters],
        )?;
        resource.isa = Some(self.library_isa(&kind)?);

        resource.validate()?;

        Ok(resource)
    }

    fn library_arch(&mut self, kind: &String) -> Result<serde_json::Value, DRRAError> {
        if let Some(cached) = self.library_cache.get(kind) {
            return Ok(cached.clone());
        }
        let lib_data = get_arch_from_library(kind, None)?;
        self.library_cache.insert(kind.clone(), lib_data.clone());
        Ok(lib_data)
    }

    fn library_isa(&mut self, kind: &String) -> Result<ComponentInstructionSet, DRRAError> {
        if let Some(cached) = self.isa_cache.get(kind) {
            return Ok(cached.clone());
        }
        let mut isa = ComponentInstructionSet::from_json(get_isa_from_library(kind, None)?)?;
        isa.component_kind = Some(kind.clone());
        self.isa_cache.insert(kind.clone(), isa.clone());
        Ok(isa)
    }
}

// The parameters a component carries: the ones its library entry requires or
// defines, plus any the file gives it -- the same set the resolver kept for it
// when the file was elaborated.
fn required_parameters(
    lib_required: &[String],
    lib_parameters: &ParameterList,
    own: &ParameterList,
) -> Vec<String> {
    let mut required: Vec<String> = Vec::new();
    for name in lib_required
        .iter()
        .chain(lib_parameters.keys())
        .chain(own.keys())
    {
        if !required.contains(name) {
            required.push(name.clone());
        }
    }
    required
}

// Each required parameter from the file if it gives it, otherwise the library
// default, otherwise the first enclosing scope that defines it.
fn fill_parameters(
    component_name: &str,
    required: &[String],
    own: &ParameterList,
    lib_parameters: &ParameterList,
    enclosing: &[&ParameterList],
) -> Result<ParameterList, DRRAError> {
    let mut parameters = ParameterList::new();
    for name in required {
        let value = own
            .get(name)
            .or_else(|| lib_parameters.get(name))
            .or_else(|| enclosing.iter().find_map(|scope| scope.get(name)));
        match value {
            Some(value) => {
                parameters.insert(name.clone(), *value);
            }
            None => {
                return Err(DRRAError::ParameterNotFound(format!(
                    "{} (in {})",
                    name, component_name
                )));
            }
        }
    }
    Ok(parameters)
}

// Give every distinct variant of a component its own name.
//
// The RTL templates instantiate a controller, resource or cell through a macro
// named after it, which a fingerprint table maps to one fingerprint per name.
// Two components sharing a name but differing in parameters -- the same
// resource template sized differently in two slots -- would both resolve to
// whichever fingerprint the table took first. Each variant beyond the first of
// a name is therefore renamed `<name>_<n>`, and the fingerprints recomputed,
// since a controller's and a resource's include its name. A fabric whose
// components all match their namesakes keeps every name it had.
fn make_names_unique(fabric: &mut Fabric) -> Result<(), DRRAError> {
    let mut taken: HashSet<String> = HashSet::new();
    for row in fabric.cells.iter() {
        for cell in row.iter() {
            taken.insert(cell.name.clone());
            if let Some(controller) = &cell.controller {
                taken.insert(controller.name.clone());
            }
            if let Some(resources) = &cell.resources {
                for resource in resources.iter() {
                    taken.insert(resource.name.clone());
                }
            }
        }
    }

    let mut controller_names = Renamer::new();
    let mut resource_names = Renamer::new();
    for row in fabric.cells.iter_mut() {
        for cell in row.iter_mut() {
            if let Some(controller) = &mut cell.controller {
                let fingerprint = controller.get_fingerprint();
                controller.name =
                    controller_names.name_for(&controller.name, &fingerprint, &mut taken);
            }
            if let Some(resources) = &mut cell.resources {
                for resource in resources.iter_mut() {
                    let fingerprint = resource.get_fingerprint();
                    resource.name =
                        resource_names.name_for(&resource.name, &fingerprint, &mut taken);
                }
            }
        }
    }
    fabric.generate_fingerprints()?;

    // A cell's fingerprint covers its controller's and resources', so cells
    // are only told apart once those are final.
    let mut cell_names = Renamer::new();
    for row in fabric.cells.iter_mut() {
        for cell in row.iter_mut() {
            let fingerprint = cell.get_fingerprint();
            cell.name = cell_names.name_for(&cell.name, &fingerprint, &mut taken);
        }
    }

    Ok(())
}

// Names handed out so far, per original name: the fingerprint each variant had
// and the name it was given.
struct Renamer {
    variants: HashMap<String, Vec<(String, String)>>,
}

impl Renamer {
    fn new() -> Self {
        Self {
            variants: HashMap::new(),
        }
    }

    fn name_for(&mut self, name: &str, fingerprint: &str, taken: &mut HashSet<String>) -> String {
        let variants = self.variants.entry(name.to_string()).or_default();
        for (known, given) in variants.iter() {
            if known == fingerprint {
                return given.clone();
            }
        }

        let given = if variants.is_empty() {
            name.to_string()
        } else {
            let mut index = variants.len();
            let mut candidate = format!("{}_{}", name, index);
            while taken.contains(&candidate) {
                index += 1;
                candidate = format!("{}_{}", name, index);
            }
            taken.insert(candidate.clone());
            candidate
        };
        variants.push((fingerprint.to_string(), given.clone()));
        given
    }
}

fn string_field(json: &serde_json::Value, key: &str, owner: &str) -> Result<String, DRRAError> {
    json.get(key)
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
        .ok_or_else(|| invalid(format!("{} has no \"{}\"", owner, key)))
}

fn bool_field(json: &serde_json::Value, key: &str) -> Option<bool> {
    json.get(key).and_then(|value| value.as_bool())
}

fn invalid(message: String) -> DRRAError {
    DRRAError::Io(Error::new(std::io::ErrorKind::InvalidInput, message))
}
