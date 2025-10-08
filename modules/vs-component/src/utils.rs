use crate::models::types::ParameterList;

use log::{debug, info, warn};
use std::{
    env, fs,
    hash::{DefaultHasher, Hasher},
    io::{Error, Result},
    path::{Path, PathBuf},
};

use bs58::encode;
use serde::Serialize;
use which::which;

pub fn get_library_path() -> Result<PathBuf> {
    let lib_path = match env::var("VESYLA_SUITE_PATH_COMPONENTS") {
        Ok(path) => PathBuf::from(path),
        Err(e) => {
            return Err(Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Environment variable VESYLA_SUITE_PATH_COMPONENTS not set: {}",
                    e
                ),
            ));
        }
    };
    // get abosulte path
    let abosulte = match std::path::absolute(lib_path) {
        Ok(path) => path,
        Err(e) => {
            return Err(Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to get absolute path: {}", e),
            ));
        }
    };

    Ok(abosulte)
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

pub fn get_path_from_library(
    component_name: &String,
    library_path: Option<&Path>,
) -> Result<PathBuf> {
    // Get library path from arg or environment variable
    let library_path = match library_path {
        Some(path) => path.to_path_buf(),
        None => match get_library_path() {
            Ok(path) => path.to_path_buf(),
            Err(e) => {
                return Err(Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Failed to get library path: {}", e),
                ));
            }
        },
    };

    // Check if a folder in the library is named the same as the cell
    for entry in walkdir::WalkDir::new(library_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_dir()
            && path
                .file_name()
                .is_some_and(|name| name == component_name.as_str())
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

pub fn get_arch_from_library(
    component_name: &String,
    library_path: Option<&Path>,
) -> Result<serde_json::Value> {
    // Check if a folder in the library is named the same as the cell
    let component_path = match get_path_from_library(component_name, library_path) {
        Ok(path) => path,
        Err(e) => {
            return Err(Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Component {} not found in the library: {}",
                    component_name, e
                ),
            ));
        }
    };

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

pub fn get_isa_from_library(
    component_name: &String,
    library_path: Option<&Path>,
) -> Result<serde_json::Value> {
    let component_path = match get_path_from_library(component_name, library_path) {
        Ok(path) => path,
        Err(e) => {
            return Err(Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Component {} not found in the library: {}",
                    component_name, e
                ),
            ));
        }
    };

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

pub fn get_rtl_files_from_library(
    component_name: &String,
    library_path: Option<&Path>,
    tmp_dir: &Path,
) -> Result<Vec<String>> {
    let component_path = match get_path_from_library(component_name, library_path) {
        Ok(path) => path,
        Err(e) => {
            return Err(e);
        }
    };
    let rtl_path = component_path.join("rtl");
    if !rtl_path.exists() {
        return Err(Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "RTL files for component \"{}\" not found in library (component path: {})",
                component_name,
                component_path.to_str().unwrap()
            ),
        ));
    }

    // Copy the component_path to a temporary directory in /tmp with random name
    let tmp_component_path = tmp_dir;
    copy_dir(&component_path, tmp_component_path)?;

    // Run the bender command to get the list of files
    let mut bender_cmd = std::process::Command::new("bender");
    let cmd = bender_cmd
        .arg("-d")
        .arg(tmp_component_path.to_str().unwrap())
        .arg("script")
        .arg("flist");
    debug!("Running bender command: {:?}", cmd);

    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) => {
            return Err(Error::other(
                format!(
                "Failed to run bender command to get the list of RTL files for component \"{}\" (component path: {}): {}",
                component_name,
                tmp_component_path.to_str().unwrap(),
                e
                ),
            ));
        }
    };

    // Check if the command was successful
    if !output.status.success() {
        return Err(Error::other(
            format!(
                "Failed to run bender command to get the list of RTL files for component \"{}\" (component path: {}): {}",
                component_name,
                tmp_component_path.to_str().unwrap(),
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }

    // Parse the output of the bender command
    let output_str = String::from_utf8_lossy(&output.stdout);
    let mut rtl_files = Vec::new();
    for line in output_str.lines() {
        rtl_files.push(line.to_string());
    }

    if rtl_files.is_empty() {
        return Err(Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Bender command did not find RTL files for component \"{}\" (component path: {})",
                component_name,
                tmp_component_path.to_str().unwrap()
            ),
        ));
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

    // Create a temporary directory for the component files
    let tmp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
    let rtl_files_list = match get_rtl_files_from_library(&kind.to_string(), None, tmp_dir.path()) {
        Ok(rtl_files) => rtl_files,
        Err(e) => {
            return Err(Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to get RTL files for component {}: {}", kind, e),
            ));
        }
    };

    debug!("RTL files: {:?}", rtl_files_list);

    for rtl_file in rtl_files_list {
        if rtl_file.is_empty() {
            continue;
        }
        // Get the name of the file from the path
        let rtl_filename = Path::new(&rtl_file).file_name().unwrap();
        // Create output file
        let output_file = output_folder.join(rtl_filename);
        debug!("Generating RTL file: {}", output_file.to_str().unwrap());

        // Get the appropriate comment prefix based on file extension
        let comment_prefix =
            if let Some(extension) = output_file.extension().and_then(|e| e.to_str()) {
                match extension {
                    "vhd" | "vhdl" => "--",
                    "sv" | "svh" | "v" | "vh" | "vt" | "verilog" | "vlg" => "//",
                    _ => "#", // Default comment for other file types
                }
            } else {
                "#" // Default if no extension
            };

        let file_comment = format!(
            "{} This file was automatically generated by Vesyla. DO NOT EDIT.\n\n",
            comment_prefix
        );

        // Check if file exists with .jinja or .j2 extension
        let template_extensions = [".jinja", ".j2"];
        let mut rtl_template_path = None;

        for ext in &template_extensions {
            let rtl_template_str = rtl_file.clone() + ext;
            let candidate_path = Path::new(&rtl_template_str);
            if candidate_path.exists() {
                rtl_template_path = Some(candidate_path.to_path_buf());
                break;
            }
        }

        if let Some(template_path) = rtl_template_path {
            let mut mj_env = minijinja::Environment::new();
            let template_dir = get_library_path()?.join("common/jinja");
            mj_env.set_loader(minijinja::path_loader(template_dir));
            mj_env.set_trim_blocks(true);
            mj_env.set_lstrip_blocks(true);
            let rtl_template_name = &template_path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            let rtl_template_content =
                fs::read_to_string(&template_path).expect("Failed to read template file");
            mj_env
                .add_template(rtl_template_name, rtl_template_content.as_str())
                .expect("Failed to add template");

            // Render the template with the component data
            let result = mj_env
                .get_template(rtl_template_name)
                .expect("Failed to get template")
                .render(serde_json::to_value(component).unwrap())
                .map_err(|e| {
                    Error::other(format!(
                        "Failed to render template for file {}: {}",
                        &template_path.display(),
                        e
                    ))
                });
            let output_str = result?;
            fs::write(&output_file, file_comment + &output_str).expect("Failed to write file");
        } else {
            // Copy the file with added comment
            let rtl_file_content = fs::read_to_string(&rtl_file).expect("Failed to read file");
            let rtl_file_content = file_comment + &rtl_file_content;
            fs::write(&output_file, rtl_file_content).expect("Failed to write file");
        }

        // Check if the file created has valid syntax
        if output_file
            .extension()
            .is_some_and(|ext| ext == "sv" || ext == "v")
        {
            check_verilog_syntax(&output_file)?;
        }
    }

    debug!(
        "Generated RTL for component {} (path: {})",
        name,
        output_folder.to_str().unwrap()
    );

    tmp_dir
        .close()
        .expect("Failed to close temporary directory");

    Ok(())
}

pub fn check_verilog_syntax(output_file: &Path) -> Result<()> {
    // Check if verible is installed
    let verible_syntax = which("verible-verilog-syntax");
    let verible_lint = which("verible-verilog-lint");
    if verible_syntax.is_err() || verible_lint.is_err() {
        warn!("Verible is not installed. Skipping syntax check for Verilog/SystemVerilog files.");
        return Ok(());
    }

    // Run verible-verilog-syntax
    let syntax_output = std::process::Command::new("verible-verilog-syntax")
        .arg(output_file.to_str().unwrap())
        .output()?;
    if !syntax_output.status.success() {
        return Err(Error::other(format!(
            "Verilog/SystemVerilog syntax check failed for file {}: {}",
            output_file.to_str().unwrap(),
            String::from_utf8_lossy(&syntax_output.stderr)
        )));
    } else {
        info!(
            "Verilog/SystemVerilog syntax check passed for file {}",
            output_file.to_str().unwrap()
        );
    }

    // Run verible-verilog-lint
    let lint_output = std::process::Command::new("verible-verilog-lint")
        .arg(output_file.to_str().unwrap())
        .output()?;
    if !String::from_utf8_lossy(&lint_output.stderr).is_empty() {
        if String::from_utf8_lossy(&lint_output.stderr).contains("error") {
            return Err(Error::other(format!(
                "Verilog/SystemVerilog lint errors for file {}: {}",
                output_file.to_str().unwrap(),
                String::from_utf8_lossy(&lint_output.stderr)
            )));
        } else {
            warn!(
                "Verilog/SystemVerilog lint warnings for file {}: {}",
                output_file.to_str().unwrap(),
                String::from_utf8_lossy(&lint_output.stderr)
            );
        }
    }

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

    encode(hash.to_be_bytes()).into_string().to_lowercase()
}

pub fn copy_rtl_dir(src: &Path, dst: &Path) -> Result<()> {
    // Create the output directory if it does not exist
    if !dst.exists() {
        fs::create_dir_all(dst).expect("Failed to create output directory");
    }
    debug!("Copying directory: {:?} to {:?}", src, dst);

    // Copy files, adding comments to Verilog/SystemVerilog files
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let target_path = dst.join(path.file_name().unwrap());

        if path.is_file() {
            let comment_content =
                "This file was automatically generated by Vesyla. DO NOT EDIT.\n\n";
            if path
                .extension()
                .is_some_and(|ext| ext == "sv" || ext == "v")
            {
                let comment = "// ".to_string() + comment_content;
                let content = fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("Failed to read file {}: {}", path.display(), e));
                let new_content = comment + &content;
                fs::write(&target_path, new_content).unwrap_or_else(|e| {
                    panic!("Failed to write file {}: {}", target_path.display(), e)
                });
            } else if path
                .extension()
                .is_some_and(|ext| ext == "vhdl" || ext == "vhd")
            {
                let comment = "-- ".to_string() + comment_content;
                let content = fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("Failed to read file {}: {}", path.display(), e));
                let new_content = comment + &content;
                fs::write(&target_path, new_content).unwrap_or_else(|e| {
                    panic!("Failed to write file {}: {}", target_path.display(), e)
                });
            } else {
                // Just copy other files
                fs::copy(&path, &target_path).unwrap_or_else(|e| {
                    panic!(
                        "Failed to copy file {} to {}: {}",
                        path.display(),
                        target_path.display(),
                        e
                    )
                });
            }
        } else if path.is_dir() {
            // For subdirectories, use the existing copy_dir function
            fs::create_dir_all(&target_path).unwrap_or_else(|e| {
                panic!(
                    "Failed to create dir for file {}: {}",
                    target_path.display(),
                    e
                )
            });
            copy_dir(&path, &target_path).unwrap_or_else(|e| {
                panic!(
                    "Failed to copy file {} to {}: {}",
                    path.display(),
                    target_path.display(),
                    e
                )
            });
        }
    }

    Ok(())
}

pub fn remove_write_permissions(dir_path: &str) -> Result<()> {
    fn set_readonly_recursive(path: &Path) -> Result<()> {
        let metadata = fs::metadata(path)?;
        let mut perms = metadata.permissions();

        // Make read-only
        // Only make specific file types read-only
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    let ext_lower = ext_str.to_lowercase();
                    // Only set read-only for json, systemverilog, vhdl and yaml files
                    if ["json", "sv", "vhdl", "vhd", "yaml", "yml"].contains(&ext_lower.as_str()) {
                        perms.set_readonly(true);
                        fs::set_permissions(path, perms)?;
                    }
                }
            }
        }

        if metadata.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                set_readonly_recursive(&entry.path())?;
            }
        }

        Ok(())
    }

    // Only change permissions of contents, not the directory itself
    let path = Path::new(dir_path);
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        set_readonly_recursive(&entry.path())?;
    }

    Ok(())
}

pub fn get_component_template_path() -> Result<PathBuf> {
    if let Ok(test_path) = env::var("VESYLA_COMPONENT_TEMPLATE_PATH") {
        return Ok(PathBuf::from(test_path));
    }

    let current_exe = env::current_exe()?;
    let current_exe_dir = current_exe
        .parent()
        .ok_or_else(|| Error::other("Failed to get parent directory of the executable"))?;
    let usr_dir = current_exe_dir.parent().ok_or_else(|| {
        Error::other("Failed to get parent directory of the executable directory")
    })?;
    let template_path = Path::new(usr_dir).join("share/vesyla/component_template");
    Ok(template_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn create_fake_library(base: PathBuf) -> Result<PathBuf> {
        let temp_dir = base;
        let library_path = temp_dir;
        let component_path = library_path.join("dummy");
        let component_rtl_path = component_path.join("rtl");
        let component_rtl_file_path = component_rtl_path.join("dummy.sv");
        fs::create_dir_all(&component_path)?;
        fs::create_dir_all(&component_rtl_path)?;
        fs::write(
            component_path.join("arch.json"),
            r#"{"parameters": {"a": 1}}"#,
        )?;
        fs::write(
            component_path.join("isa.json"),
            r#"{"parameters": {"b": 2}}"#,
        )?;
        fs::write(
            component_path.join("Bender.yml"),
            r#"
package:
  name: dummy

dependencies:

sources:
  - ./rtl/dummy.sv"#,
        )?;
        fs::write(component_rtl_file_path, r#"// This is a dummy RTL file"#)?;

        Ok(library_path)
    }

    #[test]
    fn test_get_path_from_library_not_found() {
        let result = get_path_from_library(&"nonexistent_component".to_string(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_arch_from_library_not_found() {
        let result = get_arch_from_library(&"nonexistent_component".to_string(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_isa_from_library_not_found() {
        let result = get_isa_from_library(&"nonexistent_component".to_string(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_rtl_files_from_library_not_found() {
        let tmp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
        let result =
            get_rtl_files_from_library(&"nonexistent_component".to_string(), None, tmp_dir.path());
        assert!(result.is_err());
        tmp_dir
            .close()
            .expect("Failed to close temporary directory");
    }

    #[test]
    fn test_get_arch_from_library() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
        let temp_dir_path = temp_dir.path().to_owned();
        let fake_library_path = create_fake_library(temp_dir_path).unwrap();
        let result = get_arch_from_library(&"dummy".to_string(), Some(&fake_library_path));
        assert!(result.is_ok());
        temp_dir
            .close()
            .expect("Failed to close temporary directory");
    }

    #[test]
    fn test_get_isa_from_library() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
        let temp_dir_path = temp_dir.path().to_owned();
        let fake_library_path = create_fake_library(temp_dir_path).unwrap();
        let result = get_isa_from_library(&"dummy".to_string(), Some(&fake_library_path));
        assert!(result.is_ok());
        temp_dir
            .close()
            .expect("Failed to close temporary directory");
    }

    #[test]
    fn test_get_rtl_files_from_library() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
        let temp_dir_path = temp_dir.path();
        let fake_library_path = create_fake_library(temp_dir_path.to_path_buf()).unwrap();
        assert!(
            fake_library_path.exists(),
            "Temporary library path does not exist"
        );
        assert!(
            fake_library_path.is_dir(),
            "Temporary library path is not a directory"
        );
        assert!(
            fake_library_path.join("dummy").exists(),
            "Dummy component path does not exist"
        );
        assert!(
            fake_library_path.join("dummy").is_dir(),
            "Dummy component path is not a directory"
        );
        assert!(
            fake_library_path.join("dummy").join("rtl").exists(),
            "RTL directory for dummy component does not exist"
        );
        assert!(
            fake_library_path.join("dummy").join("rtl").is_dir(),
            "RTL directory for dummy component is not a directory"
        );
        assert!(
            fake_library_path
                .join("dummy")
                .join("rtl")
                .join("dummy.sv")
                .exists(),
            "RTL file for dummy component does not exist"
        );
        let result = get_rtl_files_from_library(
            &"dummy".to_string(),
            Some(&fake_library_path),
            temp_dir_path,
        );
        assert!(
            result.is_ok(),
            "RTL files retrieval failed: {:?}",
            result.err()
        );
        temp_dir
            .close()
            .expect("Failed to close temporary directory");
    }

    #[test]
    fn test_merge_parameters_no_overlap() {
        let mut params1 = [("a".to_string(), 1u64)].iter().cloned().collect();
        let params2 = [("b".to_string(), 2u64)].iter().cloned().collect();
        let overwritten = merge_parameters(&mut params1, &params2).unwrap();
        assert_eq!(params1.get("a"), Some(&1));
        assert_eq!(params1.get("b"), Some(&2));
        assert!(overwritten.is_empty());
    }

    #[test]
    fn test_merge_parameters_with_overlap_same_value() {
        let mut params1 = [("a".to_string(), 1u64)].iter().cloned().collect();
        let params2 = [("a".to_string(), 1u64)].iter().cloned().collect();
        let overwritten = merge_parameters(&mut params1, &params2).unwrap();
        assert_eq!(params1.get("a"), Some(&1));
        assert!(overwritten.is_empty());
    }

    #[test]
    fn test_merge_parameters_with_overlap_different_value() {
        let mut params1 = [("a".to_string(), 1u64)].iter().cloned().collect();
        let params2 = [("a".to_string(), 2u64)].iter().cloned().collect();
        let overwritten = merge_parameters(&mut params1, &params2).unwrap();
        assert_eq!(params1.get("a"), Some(&1));
        assert_eq!(overwritten, vec![("a".to_string(), 1u64, 2u64)]);
    }

    #[test]
    fn test_merge_parameters_multiple() {
        let mut params1 = [("a".to_string(), 1u64), ("b".to_string(), 2u64)]
            .iter()
            .cloned()
            .collect();
        let params2 = [("b".to_string(), 3u64), ("c".to_string(), 4u64)]
            .iter()
            .cloned()
            .collect();
        let overwritten = merge_parameters(&mut params1, &params2).unwrap();
        assert_eq!(params1.get("a"), Some(&1));
        assert_eq!(params1.get("b"), Some(&2));
        assert_eq!(params1.get("c"), Some(&4));
        assert_eq!(overwritten, vec![("b".to_string(), 2u64, 3u64)]);
    }

    #[test]
    fn test_get_parameters_array() {
        let json = serde_json::json!({
            "parameters": [
                { "name": "foo", "value": 42 },
                { "name": "bar", "value": 7 }
            ]
        });
        let params = get_parameters(&json, None);
        assert_eq!(params.get("foo"), Some(&42));
        assert_eq!(params.get("bar"), Some(&7));
    }

    #[test]
    fn test_get_parameters_object() {
        let json = serde_json::json!({
            "parameters": {
                "foo": 123,
                "bar": 456
            }
        });
        let params = get_parameters(&json, None);
        assert_eq!(params.get("foo"), Some(&123));
        assert_eq!(params.get("bar"), Some(&456));
    }

    #[test]
    fn test_get_parameters_custom_key() {
        let json = serde_json::json!({
            "custom": {
                "baz": 99
            }
        });
        let params = get_parameters(&json, Some("custom".to_string()));
        assert_eq!(params.get("baz"), Some(&99));
    }

    #[test]
    fn test_get_parameters_missing() {
        let json = serde_json::json!({});
        let params = get_parameters(&json, None);
        assert!(params.is_empty());
    }

    #[test]
    fn test_generate_hash_different_names() {
        let params = BTreeMap::new();
        let hash1 = generate_hash(vec!["foo".to_string()], &params);
        let hash2 = generate_hash(vec!["bar".to_string()], &params);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_generate_hash_different_params() {
        let mut params1 = BTreeMap::new();
        params1.insert("a".to_string(), 1u64);
        let mut params2 = BTreeMap::new();
        params2.insert("a".to_string(), 2u64);
        let hash1 = generate_hash(vec!["foo".to_string()], &params1);
        let hash2 = generate_hash(vec!["foo".to_string()], &params2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_generate_hash_same_input_same_output() {
        let mut params = BTreeMap::new();
        params.insert("x".to_string(), 42u64);
        let hash1 = generate_hash(vec!["baz".to_string()], &params);
        let hash2 = generate_hash(vec!["baz".to_string()], &params);
        assert_eq!(hash1, hash2);
    }
}
