use std::{
    env,
    io::{Error, Result},
    path::{Path, PathBuf},
};

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
