use log::{debug, warn};
use std::env;
use std::io::{Error, Result};
use std::path::PathBuf;

pub fn get_library_path() -> String {
    let lib_path = env::var("VESYLA_LIBRARY_PATH").expect("Environment variable VESYLA_LIBRARY_PATH not set! Did you forget to source the setup script env.sh?");
    // get abosulte path
    let abosulte = std::path::absolute(lib_path).expect("Cannot get absolute path for library");
    abosulte.to_str().unwrap().to_string()
}

pub fn get_path_from_library(component_name: &String) -> Result<PathBuf> {
    let library_path = get_library_path();
    debug!("Library path: {}", library_path);
    debug!("Looking for component: {}", component_name);
    // Check if a folder in the library is named the same as the cell
    for entry in walkdir::WalkDir::new(&library_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_dir()
            && path
                .file_name()
                .map_or(false, |name| name == component_name.as_str())
        {
            debug!(
                "Component {} found in the library at path {}",
                component_name,
                path.to_str().unwrap()
            );
            return Ok(entry.path().to_path_buf());
        }
    }

    Err(Error::new(
        std::io::ErrorKind::NotFound,
        format!("Component {} not found in the library", component_name),
    ))
}

pub fn get_arch_from_library(component_name: &String) -> Result<serde_json::Value> {
    // Check if a folder in the library is named the same as the cell
    let component_path = get_path_from_library(component_name)?;

    // Get the arch.json file for the cell
    let arch_path = component_path.join("arch.json");
    if !arch_path.exists() {
        warn!(
            "Component \"{}\" JSON description not found in library (component path: {})",
            component_name,
            component_path.to_str().unwrap()
        );
        return Err(Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Component {} does not contain an arch.json file",
                component_name
            ),
        ));
    }

    // Read the arch.json file
    let json_str = std::fs::read_to_string(&arch_path).expect("Failed to read file");
    let component_result = serde_json::from_str(&json_str);
    match component_result {
        Ok(component) => Ok(component),
        Err(_) => {
            warn!(
                "Failed to parse JSON description for component \"{}\" (component path: {})",
                component_name,
                arch_path.to_str().unwrap()
            );
            Err(Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse json file: {}", arch_path.to_str().unwrap()),
            ))
        }
    }
}

pub fn get_rtl_template_from_library(
    component_name: &String,
    template_name: Option<String>,
) -> Result<String> {
    let template_name = template_name.unwrap_or("rtl.sv".to_string()).clone();
    debug!(
        "Looking for RTL template for component \"{}\" (template: {})",
        component_name, template_name
    );
    let component_path = get_path_from_library(component_name)?;
    // Check if a folder in the library is named the same as the cell
    let rtl_path = component_path.join(template_name);
    if !rtl_path.exists() {
        warn!(
            "RTL template for component \"{}\" not found in library (component path: {})",
            component_name,
            component_path.to_str().unwrap()
        );
        return Err(Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Component {} does not contain an rtl folder",
                component_name
            ),
        ));
    }
    debug!(
        "RTL template found for component \"{}\" (path: {})",
        component_name,
        rtl_path.to_str().unwrap()
    );

    // Read the rtl.sv file
    let rtl_str = std::fs::read_to_string(&rtl_path).expect("Failed to read file");
    Ok(rtl_str)
}
