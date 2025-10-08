use crate::models::isa::InstructionSet;
use std::{fs, path::Path};

pub trait ISAGenerator {
    fn generate_markdown(&self, output_path: &Path) -> Result<(), std::io::Error>;
    fn generate_json(&self, output_path: &Path) -> Result<(), std::io::Error>;
}

impl ISAGenerator for InstructionSet {
    fn generate_markdown(&self, output_path: &Path) -> Result<(), std::io::Error> {
        // using minijinja to generate the ISA documentation
        let template = include_str!("../assets/isa_doc_template.md.jinja");
        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_template("doc", template).unwrap();
        let tmpl = env.get_template("doc").unwrap();
        let isa = self;
        let result = tmpl.render(minijinja::context!(isa)).unwrap();

        // write the result to the output file
        let output_path = output_path.join("isa.md");
        // if the output directory does not exist, create it
        match std::fs::write(output_path, result) {
            Ok(_) => Ok(()),
            Err(e) => {
                eprintln!("Failed to write documentation to file: {}", e);
                Err(std::io::Error::other("Failed to write documentation"))
            }
        }
    }

    fn generate_json(&self, output_path: &Path) -> Result<(), std::io::Error> {
        // if output directory does not exist, create it
        let output_dir = std::path::Path::new(output_path).parent().unwrap();
        if !output_dir.exists() {
            fs::create_dir_all(output_dir).expect("Failed to create output directory");
        }

        // write the result to the output file
        let output_path = output_path.join("isa.json");
        let output_file = fs::File::create(&output_path)?;
        serde_json::to_writer_pretty(output_file, self)?;

        Ok(())
    }
}
