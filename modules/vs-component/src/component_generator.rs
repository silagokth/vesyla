use serde_json::json;
use std::{
    fs::{read_to_string, File},
    io::{Error, Result, Write},
    path::Path,
};

use crate::utils::get_component_template_path;

pub struct ComponentGenerator {}

impl ComponentGenerator {
    pub fn create(arch_json: &Path, isa_json: &Path, output_dir: &Path, force: bool) -> Result<()> {
        let arch_json_value = json!(&read_to_string(arch_json)?);
        let isa_json_value = json!(&read_to_string(isa_json)?);

        Self::validate_arch_json(&arch_json_value)?;
        Self::validate_isa_json(&isa_json_value)?;

        let combined_json = Self::get_combined_json(&arch_json_value, &isa_json_value)?;

        let template_path = get_component_template_path()?;

        Self::copy_and_render_files(template_path.as_path(), &combined_json, output_dir, force)?;

        Ok(())
    }

    fn validate_arch_json(arch_json: &serde_json::Value) -> Result<()> {
        let schema_str = include_str!("../assets/component_arch_schema.json");
        Self::validate_json(arch_json, &json!(schema_str.as_bytes()))
    }

    fn validate_isa_json(isa_json: &serde_json::Value) -> Result<()> {
        let schema_str = include_str!("../assets/component_isa_schema.json");
        Self::validate_json(isa_json, &json!(schema_str.as_bytes()))
    }

    fn validate_json(instance: &serde_json::Value, schema: &serde_json::Value) -> Result<()> {
        jsonschema::validate(schema, instance).map_err(|e| {
            Error::new(
                std::io::ErrorKind::InvalidData,
                format!("JSON validation error: {}", e),
            )
        })
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

    fn get_combined_json(
        arch_json: &serde_json::Value,
        isa_json: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let mut combined_json = arch_json.clone();
        if let Some(obj) = combined_json.as_object_mut() {
            obj.insert("isa".to_string(), isa_json.clone());
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

        for entry in std::fs::read_dir(template_path)? {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name();
            let dest_path = output_dir.join(file_name);

            if path.is_file() {
                if path.extension().and_then(|s| s.to_str()) == Some("jinja") {
                    let dest_path = dest_path.with_extension("");
                    Self::copy_and_render_file(&path, &dest_path, context)?;
                } else {
                    std::fs::copy(&path, &dest_path)?;
                }
            } else if path.is_dir() {
                Self::copy_and_render_files(&path, context, &dest_path, force)?;
            }
        }
        Ok(())
    }

    fn copy_and_render_file(src: &Path, dest: &Path, context: &serde_json::Value) -> Result<()> {
        let template = read_to_string(src)?;
        let mut environment = minijinja::Environment::new();
        environment
            .add_template("template", template.as_str())
            .map_err(|e| {
                Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to add template: {}", e),
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

        let comment = Self::get_comment_string(src.with_extension("").extension())?;

        let result_string = result.map_err(|e| {
            Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to render template: {}", e),
            )
        })?;
        let output = comment + result_string.as_str();
        let mut file = File::create(dest)?;
        Write::write_all(&mut file, output.as_bytes())?;

        Ok(())
    }

    fn get_comment_string(file_extension: Option<&std::ffi::OsStr>) -> Result<String> {
        let mut comment =
            "This file was automatically generated by Vesyla. DO NOT EDIT.\n\n".to_string();
        match file_extension.and_then(|s| s.to_str()) {
            Some("rs") => comment = "// ".to_string() + &comment,
            Some("h") | Some("hpp") => comment = "// ".to_string() + &comment,
            Some("cpp") => comment = "// ".to_string() + &comment,
            Some("yml") | Some("yaml") => comment = "# ".to_string() + &comment,
            _ => {}
        }

        Ok(comment)
    }
}
