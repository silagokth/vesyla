use std::{
    fs::{read_to_string, File},
    io::{stdin, stdout, Error, Result, Write},
    path::Path,
};

use crate::utils::get_component_template_path;

pub struct ComponentGenerator {}

impl ComponentGenerator {
    pub fn create(
        arch_json: &Path,
        isa_json: &Path,
        output_dir: &Path,
        force: bool,
        non_interactive: bool,
    ) -> Result<()> {
        let arch_json_value: serde_json::Value = serde_json::from_str(&read_to_string(arch_json)?)?;
        let isa_json_value: serde_json::Value = serde_json::from_str(&read_to_string(isa_json)?)?;

        Self::validate_arch_json(&arch_json_value)?;
        log::info!("Validated architecture JSON file: {}", arch_json.display());
        Self::validate_isa_json(&isa_json_value)?;
        log::info!("Validated ISA JSON file: {}", isa_json.display());

        let combined_json =
            Self::get_combined_json(&arch_json_value, &isa_json_value, non_interactive)?;

        let template_path = get_component_template_path()?;

        Self::copy_and_render_files(template_path.as_path(), &combined_json, output_dir, force)?;

        // Copy JSON files to output directory
        std::fs::create_dir_all(output_dir)?;
        std::fs::copy(arch_json, output_dir.join(arch_json.file_name().unwrap()))?;
        std::fs::copy(isa_json, output_dir.join(isa_json.file_name().unwrap()))?;

        log::info!(
            "Generated component files in output directory: {}",
            output_dir.display()
        );

        Ok(())
    }

    fn validate_arch_json(arch_json: &serde_json::Value) -> Result<()> {
        let arch_schema_str = include_str!("../assets/component_arch_schema.json");
        Self::validate_json(arch_json, &serde_json::from_str(arch_schema_str)?)
    }

    fn validate_isa_json(isa_json: &serde_json::Value) -> Result<()> {
        let isa_schema_str = include_str!("../assets/component_isa_schema.json");
        Self::validate_json(isa_json, &serde_json::from_str(isa_schema_str)?)
    }

    fn validate_json(instance: &serde_json::Value, schema: &serde_json::Value) -> Result<()> {
        let validator = jsonschema::validator_for(schema).map_err(|e| {
            Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to create JSON schema validator: {}", e),
            )
        })?;
        validator.validate(instance).map_err(|e| {
            Error::new(
                std::io::ErrorKind::InvalidData,
                format!("JSON validation error: {}", e),
            )
        })?;

        Ok(())
    }

    fn get_component_kind(arch_json: &serde_json::Value) -> Result<String> {
        if let Some(kind) = arch_json.get("kind") {
            if let Some(kind_str) = kind.as_str() {
                return Ok(kind_str.to_string());
            }
        }
        Err(Error::new(
            std::io::ErrorKind::InvalidData,
            "component_kind not found in architecture JSON",
        ))
    }

    fn get_component_type(arch_json: &serde_json::Value) -> Result<String> {
        if let Some(comp_type) = arch_json.get("type") {
            if let Some(type_str) = comp_type.as_str() {
                return Ok(type_str.to_string());
            }
        }
        Err(Error::new(
            std::io::ErrorKind::InvalidData,
            "component type not found in architecture JSON",
        ))
    }

    fn get_combined_json(
        arch_json: &serde_json::Value,
        isa_json: &serde_json::Value,
        non_interactive: bool,
    ) -> Result<serde_json::Value> {
        let mut combined_json = arch_json.clone();
        if let Some(obj) = combined_json.as_object_mut() {
            obj.insert("isa".to_string(), isa_json.clone());
            if obj.get("category").is_none() {
                if non_interactive {
                    obj.insert(
                        "category".to_string(),
                        serde_json::Value::String("uncategorized".to_string()),
                    );
                }
                let mut user_input = String::new();
                print!("Specify the component type: '0: uncategorized', '1: processor', '2: memory', '3: network' or '4: system': ");
                stdout().flush().unwrap();
                stdin().read_line(&mut user_input).map_err(|e| {
                    Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Failed to read user input: {}", e),
                    )
                })?;
                match user_input.trim().to_lowercase().as_str() {
                    "0" | "uncategorized" => {
                        obj.insert(
                            "category".to_string(),
                            serde_json::Value::String("uncategorized".to_string()),
                        );
                    }
                    "1" | "processor" => {
                        obj.insert(
                            "category".to_string(),
                            serde_json::Value::String("processor".to_string()),
                        );
                    }
                    "2" | "memory" => {
                        obj.insert(
                            "category".to_string(),
                            serde_json::Value::String("memory".to_string()),
                        );
                    }
                    "3" | "network" => {
                        obj.insert(
                            "category".to_string(),
                            serde_json::Value::String("network".to_string()),
                        );
                    }
                    "4" | "system" => {
                        obj.insert(
                            "category".to_string(),
                            serde_json::Value::String("system".to_string()),
                        );
                    }
                    _ => {
                        return Err(Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid component category. Please enter 0: 'uncategorized', 1: 'processor', 2: 'memory', 3: 'network' or 4: 'system'.",
                        ));
                    }
                }
            }
            if obj.get("type").is_none() {
                if non_interactive {
                    return Err(Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Component type is not specified in architecture JSON. In non-interactive mode, please specify 'type' as 'controller' or 'resource'.",
                    ));
                }
                let mut user_input = String::new();
                print!("Specify the component type: '0: controller' or '1: resource': ");
                stdout().flush().unwrap();
                stdin().read_line(&mut user_input).map_err(|e| {
                    Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Failed to read user input: {}", e),
                    )
                })?;
                match user_input.trim().to_lowercase().as_str() {
                    "1" | "resource" => {
                        obj.insert(
                            "type".to_string(),
                            serde_json::Value::String("resource".to_string()),
                        );
                    }
                    "0" | "controller" => {
                        obj.insert(
                            "type".to_string(),
                            serde_json::Value::String("controller".to_string()),
                        );
                    }
                    _ => {
                        return Err(Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid component type. Please enter 0: 'resource' or 1: 'controller'.",
                        ));
                    }
                }
            }
        } else {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Architecture JSON is not an object",
            ));
        }
        Ok(combined_json.clone())
    }

    fn copy_and_render_files(
        template_path: &Path,
        context: &serde_json::Value,
        output_dir: &Path,
        force: bool,
    ) -> Result<()> {
        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)?;
        } else if !force {
            return Err(Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "Output directory '{}' already exists. Use force (-f) option to overwrite.",
                    output_dir.display()
                ),
            ));
        }

        let template_folder = std::fs::read_dir(template_path).map_err(|e| {
            Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Failed to read template directory '{}': {}",
                    template_path.display(),
                    e
                ),
            )
        })?;
        for entry in template_folder {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name();
            let dest_path = output_dir.join(file_name);

            if path.is_file() {
                if path.extension().and_then(|s| s.to_str()) == Some("jinja") {
                    let dest_path = dest_path.with_extension("");
                    Self::copy_and_render_file(&path, &dest_path, context)?;
                } else {
                    let mut new_filename = dest_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("output")
                        .to_string();
                    let component_kind = Self::get_component_kind(context)?;
                    if component_kind.contains("component") {
                        return Err(Error::new(
                            std::io::ErrorKind::InvalidData,
                            "component_kind contains 'component', which is reserved. Please specify a different kind.",
                        ));
                    }
                    new_filename = new_filename.replace("component", component_kind.as_str());
                    let dest_path = dest_path
                        .with_file_name(new_filename)
                        .with_extension("sv.j2");
                    std::fs::copy(&path, &dest_path)?;
                }
            } else if path.is_dir() {
                let component_type = Self::get_component_type(context)?;
                if component_type == "controller"
                    && path.file_name().and_then(|s| s.to_str()) == Some("compile_util")
                {
                    continue;
                } else {
                    Self::copy_and_render_files(&path, context, &dest_path, force)?;
                }
            }
        }
        Ok(())
    }

    fn copy_and_render_file(src: &Path, dest: &Path, context: &serde_json::Value) -> Result<()> {
        let template = read_to_string(src)?;
        let mut environment = minijinja::Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        environment
            .add_template("template", template.as_str())
            .map_err(|e| {
                Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to add template {}: {}", src.display(), e),
                )
            })?;
        let jinja_template = environment.get_template("template").map_err(|e| {
            Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to get template: {}", e),
            )
        })?;
        let result = jinja_template.render(context);

        // if name contains "component", replace it with component_kind
        if let Some(file_name) = dest.file_name().and_then(|s| s.to_str()) {
            if file_name.contains("component") {
                let component_kind = Self::get_component_kind(context)?;
                if component_kind.contains("component") {
                    return Err(Error::new(
                        std::io::ErrorKind::InvalidData,
                        "component_kind contains 'component', which is reserved. Please specify a different kind.",
                    ));
                }
                let new_file_name = file_name.replace("component", &component_kind);
                let new_dest = dest.with_file_name(new_file_name);
                return Self::copy_and_render_file(src, &new_dest, context);
            }
        }

        let result_string = result.map_err(|e| {
            Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to render template {}: {}", src.display(), e),
            )
        })?;
        let output = result_string.as_str();
        let mut file = File::create(dest)?;
        Write::write_all(&mut file, output.as_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::tempdir;

    #[test]
    fn test_create_component() {
        env::set_var("RUST_LOG", "debug");
        env::set_var("VESYLA_COMPONENT_TEMPLATE_PATH", "./template");

        let mut arch_json =
            serde_json::from_str::<serde_json::Value>(include_str!("../tests/component_arch.json"))
                .unwrap();
        arch_json.as_object_mut().unwrap().insert(
            "category".to_string(),
            serde_json::Value::String("processor".to_string()),
        );
        arch_json.as_object_mut().unwrap().insert(
            "type".to_string(),
            serde_json::Value::String("resource".to_string()),
        );

        let arch_json = serde_json::to_string_pretty(&arch_json).unwrap();
        let isa_json = include_str!("../tests/component_isa.json");

        let temp_dir = tempdir().unwrap();
        let arch_path = temp_dir.path().join("arch.json");
        let isa_path = temp_dir.path().join("isa.json");
        let output_dir = temp_dir.path().join("output");

        std::fs::write(&arch_path, arch_json).unwrap();
        std::fs::write(&isa_path, isa_json).unwrap();

        ComponentGenerator::create(&arch_path, &isa_path, &output_dir, true, true).unwrap();

        assert!(output_dir.exists());

        let rtl_output_path = output_dir.join("rtl");
        let sst_output_path = output_dir.join("sst");
        let compile_output_path = output_dir.join("compile_util");

        // Check the j2 files were not changed
        assert!(rtl_output_path.join("dpu.sv.j2").exists());
        assert!(rtl_output_path.join("dpu_pkg.sv.j2").exists());

        // Check the sst headers were generated
        assert!(sst_output_path.join("dpu.h").exists());
        assert!(sst_output_path.join("dpu.cpp").exists());
        assert!(sst_output_path.join("dpu_pkg.h").exists());
        assert!(sst_output_path.join("dpu_pkg.cpp").exists());

        // Check the compile_util files were generated
        assert!(compile_output_path.join("src").join("main.rs").exists());

        // Check Bender file was generated
        assert!(output_dir.join("Bender.yml").exists());

        env::remove_var("VESYLA_COMPONENT_TEMPLATE_PATH");
        let var = env::var("VESYLA_COMPONENT_TEMPLATE_PATH");
        assert!(var.is_err());

        assert!(temp_dir.close().is_ok());
    }
}
