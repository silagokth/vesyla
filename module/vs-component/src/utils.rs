use crate::drra::ParameterList;
use bs58::encode;
use log::{debug, warn};
use rand::Rng;
use serde::Serialize;
use std::hash::{DefaultHasher, Hasher};
use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::{env, fs};

pub fn get_library_path() -> String {
    let lib_path = env::var("VESYLA_SUITE_PATH_COMPONENTS").expect("Environment variable VESYLA_SUITE_PATH_COMPONENTS not set! Did you forget to source the setup script env.sh?");
    // get abosulte path
    let abosulte = std::path::absolute(lib_path).expect("Cannot get absolute path for library");
    abosulte.to_str().unwrap().to_string()
}

pub fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let path = entry.path();
        let dst_path = dst.join(path.file_name().unwrap());

        if ty.is_dir() {
            copy_dir(&path, &dst_path)?;
        } else {
            fs::copy(&path, &dst_path)?;
        }
    }

    Ok(())
}

pub fn get_parameters(component: &serde_json::Value, param_key: Option<String>) -> ParameterList {
    let param_key = param_key.unwrap_or("parameters".to_string());
    let mut parameters = ParameterList::new();
    let component_params = component.get(param_key);
    if let Some(component_params) = component_params {
        // check if the parameters are an array
        if component_params.is_array() {
            for param in component_params.as_array().unwrap() {
                let name = param.get("name").unwrap().as_str().unwrap();
                let value = param.get("value").unwrap();
                parameters.insert(name.to_string(), value.as_u64().unwrap());
            }
        } else {
            // check if the parameters are an object
            for (name, value) in component_params.as_object().unwrap() {
                parameters.insert(name.to_string(), value.as_u64().unwrap());
            }
        }
    }
    parameters
}

/// Merge parameters from params2 into params1
pub fn merge_parameters(
    params1: &mut ParameterList,
    params2: &ParameterList,
) -> Result<Vec<(String, u64, u64)>> {
    let mut overwritten_params = Vec::new();
    for (param_name, param_value) in params2.iter() {
        // Check if the parameter already exists in params1
        if !params1.contains_key(param_name) {
            params1.insert(param_name.clone(), *param_value);
        } else {
            let existing_param = params1.get(param_name).unwrap();
            if existing_param != param_value {
                // List the parameters that exist in param1 but with a different value
                overwritten_params.push((param_name.clone(), *existing_param, *param_value));
            }
        }
    }
    Ok(overwritten_params)
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

pub fn get_isa_from_library(component_name: &String) -> Result<serde_json::Value> {
    let component_path = get_path_from_library(component_name)?;

    // Get the isa.json file for the cell
    let isa_path = component_path.join("isa.json");

    if !isa_path.exists() {
        warn!(
            "Component \"{}\" JSON description not found in library (component path: {})",
            component_name,
            component_path.to_str().unwrap()
        );
        Err(Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Component {} does not contain an isa.json file",
                component_name
            ),
        ))
    } else {
        let json_str = std::fs::read_to_string(&isa_path).expect("Failed to read file");
        serde_json::from_str(&json_str).map_err(|_| {
            warn!(
                "Failed to parse JSON description for component \"{}\" (component path: {})",
                component_name,
                isa_path.to_str().unwrap()
            );
            Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse json file: {}", isa_path.to_str().unwrap()),
            )
        })
    }
}

pub fn get_rtl_files_from_library(component_name: &String) -> Result<Vec<String>> {
    let component_path = get_path_from_library(component_name)?;
    let rtl_path = component_path.join("rtl");
    if !rtl_path.exists() {
        warn!(
            "RTL files for component \"{}\" not found in library (component path: {})",
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

    // Copy the component_path to a temporary directory in /tmp with random name
    let mut rng = rand::thread_rng();
    let random_folder_name = format!("{}_{}", component_name, rng.gen::<u32>());
    let tmp_path_str = env::var("VESYLA_SUITE_PATH_TMP")
        .expect("Environment variable VESYLA_SUITE_PATH_TMP not set!");
    let tmp_dir = Path::new(&tmp_path_str).join(random_folder_name);
    copy_dir(&component_path, &tmp_dir).expect("Failed to copy directory");

    let tmp_component_path = tmp_dir;

    // Run the bender command to get the list of files
    let mut bender_cmd = std::process::Command::new("bender");
    let cmd = bender_cmd
        .arg("-d")
        .arg(tmp_component_path.to_str().unwrap())
        .arg("script")
        .arg("flist");

    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) => {
            warn!(
                "Failed to run bender command to get the list of RTL files for component \"{}\" (component path: {}): {}",
                component_name,
                tmp_component_path.to_str().unwrap(),
                e
            );
            return Err(Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "Failed to run bender command to get the list of RTL files for component {}",
                    component_name
                ),
            ));
        }
    };

    // Check if the command was successful
    if !output.status.success() {
        warn!(
            "Failed to run bender command to get the list of RTL files for component \"{}\" (component path: {}): {}",
            component_name,
            tmp_component_path.to_str().unwrap(),
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(Error::new(
            std::io::ErrorKind::Other,
            format!(
                "Failed to run bender command to get the list of RTL files for component {}",
                component_name
            ),
        ));
    }

    // Parse the output of the bender command
    let output_str = String::from_utf8_lossy(&output.stdout);
    let mut rtl_files = Vec::new();
    for line in output_str.lines() {
        rtl_files.push(line.to_string());
    }

    Ok(rtl_files)
}

pub fn generate_rtl_for_component(
    kind: &str,
    name: &str,
    output_folder: &Path,
    component: &impl Serialize,
) -> Result<()> {
    // Check if output folder exists
    if !output_folder.exists() {
        fs::create_dir_all(output_folder)?;
    } else {
        warn!(
            "Overwriting existing RTL for controller {} (path: {})",
            name,
            output_folder.to_str().unwrap()
        );
    }

    let rtl_files_list = get_rtl_files_from_library(&kind.to_string()).expect(&format!(
        "Failed to get RTL files from library for {}",
        kind
    ));
    debug!("RTL files: {:?}", rtl_files_list);

    for rtl_file in rtl_files_list {
        if rtl_file.is_empty() {
            continue;
        }
        // Get the name of the file from the path
        let rtl_filename = Path::new(&rtl_file).file_name().unwrap();
        // Create output file
        let output_file = output_folder.join(&rtl_filename);
        debug!("Generating RTL file: {}", output_file.to_str().unwrap());
        // Check if file exists with .j2 extension
        let rtl_template_str = rtl_file.clone() + ".j2";
        let rtl_template_path = Path::new(&rtl_template_str);
        if rtl_template_path.exists() {
            let mut mj_env = minijinja::Environment::new();
            mj_env.set_trim_blocks(true);
            mj_env.set_lstrip_blocks(true);
            let rtl_template_content =
                fs::read_to_string(&rtl_template_path).expect("Failed to read template file");
            mj_env
                .add_template("rtl_template", &rtl_template_content.as_str())
                .expect("Failed to add template");
            let result = mj_env
                .get_template("rtl_template")
                .expect("Failed to get template")
                .render(&component);
            let output_str = result.expect("Failed to render template");
            fs::write(&output_file, output_str).expect("Failed to write file");
        } else {
            // Copy the file
            fs::copy(Path::new(&rtl_file), output_file).expect("Failed to copy file");
        }
    }

    debug!(
        "Generated RTL for component {} (path: {})",
        name,
        output_folder.to_str().unwrap()
    );

    Ok(())
}

pub fn generate_hash(names: Vec<String>, parameters: &ParameterList) -> String {
    let mut hasher = DefaultHasher::new();
    for name in names {
        hasher.write(name.as_bytes());
    }
    for (param_name, param_value) in parameters.iter() {
        hasher.write(param_name.as_bytes());
        hasher.write(&param_value.to_be_bytes());
    }
    let hash = hasher.finish();

    let str_hash = encode(hash.to_be_bytes()).into_string().to_lowercase();
    "_".to_string() + &str_hash
}
