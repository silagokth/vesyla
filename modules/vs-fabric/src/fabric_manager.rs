use crate::isa_gen::ISAGenerator;
use crate::loader::FabricLoader;
use crate::models::drra::Fabric;
use crate::resolver::HierarchicalResolver;
use crate::rtl_generator::RTLGenerator;
use crate::utils::{copy_rtl_dir, get_path_from_library, remove_write_permissions};
use crate::{arch_visual_gen, sst_sim_gen};

use log::{debug, error, info};
use std::{
    fs,
    io::Error,
    path::{Path, PathBuf},
};

// The fabric is built in three steps, each writing its own subdirectory of the
// output directory:
//
//   elaborate  source arch.json   -> arch/ (elaborated arch.json, visualization), isa/
//   sst        elaborated or sized arch.json -> sst/
//   rtl        elaborated or sized arch.json -> rtl/
//
// The compiler only needs what elaborate writes. sst and rtl come after it, so
// they can be built from the sized arch.json the compiler writes for a program.
pub struct FabricManager {
    output_dir: PathBuf,
}

impl FabricManager {
    pub fn new(output_dir: &str) -> Result<Self, Error> {
        let output_dir = Path::new(output_dir).to_path_buf();
        fs::create_dir_all(&output_dir)?;
        Ok(Self { output_dir })
    }

    pub fn elaborate(&self, arch_json_path: &Path) -> Result<(), Error> {
        info!("Elaborating the fabric...");
        let arch_output_dir = self.output_dir.join("arch");
        let isa_output_dir = self.output_dir.join("isa");
        fs::create_dir_all(&arch_output_dir)?;
        fs::create_dir_all(&isa_output_dir)?;

        let mut resolver = HierarchicalResolver::new();
        let mut alimp = resolver.resolve_alimp(arch_json_path).map_err(|e| {
            Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Resolution failed: {}", e),
            )
        })?;

        let arch_output_file = arch_output_dir.join("arch.json");
        self.write_fabric_json(alimp.drra.as_mut().unwrap(), &arch_output_file)?;

        let isa = &alimp.get_isa().map_err(|e| {
            Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Failed to get ISA: {}", e),
            )
        })?;
        isa.generate_json(&isa_output_dir)?;
        isa.generate_markdown(&isa_output_dir)?;

        info!("Generating architecture visualization...");
        arch_visual_gen::generate(&arch_output_file, &arch_output_dir);

        finalize(&arch_output_dir)?;
        finalize(&isa_output_dir)?;

        info!("Elaboration complete!");
        Ok(())
    }

    pub fn generate_sst(&self, arch_json_path: &Path) -> Result<(), Error> {
        info!("Generating SST simulation files...");
        let sst_output_dir = self.output_dir.join("sst");
        fs::create_dir_all(&sst_output_dir)?;

        let fabric = load_fabric(arch_json_path)?;
        let arch = serde_json::to_value(&fabric)?;
        sst_sim_gen::generate(&arch, &sst_output_dir);

        finalize(&sst_output_dir)?;

        info!("SST generation complete!");
        Ok(())
    }

    pub fn generate_rtl(&self, arch_json_path: &Path) -> Result<(), Error> {
        info!("Generating RTL...");
        let rtl_output_dir = self.output_dir.join("rtl");
        fs::create_dir_all(&rtl_output_dir)?;

        let mut fabric = load_fabric(arch_json_path)?;
        let mut rtl_generator = RTLGenerator::new(&rtl_output_dir);
        rtl_generator.generate(&mut fabric)?;

        info!("Copying shared artifacts...");
        copy_common_files(&rtl_output_dir)?;
        copy_testbench_files(&rtl_output_dir)?;

        finalize(&rtl_output_dir)?;

        info!("RTL generation complete!");
        Ok(())
    }

    fn write_fabric_json(&self, fabric: &Fabric, output_path: &Path) -> Result<(), Error> {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = fs::File::create(output_path)?;
        serde_json::to_writer_pretty(file, fabric)?;

        info!("Generated architecture JSON: {}", output_path.display());
        Ok(())
    }
}

fn load_fabric(arch_json_path: &Path) -> Result<Fabric, Error> {
    let mut loader = FabricLoader::new();
    loader.load(arch_json_path).map_err(|e| {
        Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Failed to load {}: {}", arch_json_path.display(), e),
        )
    })
}

fn copy_common_files(rtl_output_dir: &Path) -> Result<(), Error> {
    let common_dir = get_path_from_library(&"common".to_string(), None).map_err(|e| {
        Error::new(
            std::io::ErrorKind::NotFound,
            format!("Common library not found: {}", e),
        )
    })?;

    for entry in fs::read_dir(common_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let rtl_path = path.join("rtl");
            let bender_yml = path.join("Bender.yml");

            if bender_yml.exists() {
                debug!("Found Bender.yml in directory: {:?}", path);

                // Create output directory structure
                let component_output_dir = rtl_output_dir
                    .join("common")
                    .join(path.file_name().unwrap());
                fs::create_dir_all(&component_output_dir)?;

                // Copy Bender file with header
                let bender_output = component_output_dir.join("Bender.yml");
                let header = "# This file was automatically generated by Vesyla. DO NOT EDIT.\n\n";
                let content = header.to_string() + &fs::read_to_string(&bender_yml)?;
                fs::write(&bender_output, content)?;

                // Copy RTL files
                let rtl_output_dir = component_output_dir.join("rtl");
                copy_rtl_dir(&rtl_path, &rtl_output_dir)?;

                debug!("Copied common component: {:?}", path.file_name().unwrap());
            }
        }
    }

    Ok(())
}

fn copy_testbench_files(rtl_output_dir: &Path) -> Result<(), Error> {
    let testbench_dir = get_path_from_library(&"tb".to_string(), None)?;
    let tb_output_dir = rtl_output_dir.join("tb");

    fs::create_dir_all(&tb_output_dir)?;
    copy_rtl_dir(&testbench_dir, &tb_output_dir)?;

    debug!("Copied testbench files");
    Ok(())
}

// Make what a step wrote read-only. Only the step's own subdirectory, so the
// steps that follow can still write theirs next to it.
fn finalize(step_output_dir: &Path) -> Result<(), Error> {
    remove_write_permissions(&step_output_dir.to_string_lossy()).map_err(|e| {
        error!("Failed to remove write permissions: {}", e);
        Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("Failed to finalize {}: {}", step_output_dir.display(), e),
        )
    })
}
