use std::path::PathBuf;
use std::process::Command;

pub fn set_git_version_env(var_name: &str) {
    let git_dir_output = Command::new("git")
        .args(&["rev-parse", "--git-dir"])
        .output();

    let git_dir = match git_dir_output {
        Ok(output) if output.status.success() => {
            let dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if dir.is_empty() {
                None
            } else {
                Some(dir)
            }
        }
        _ => None,
    };

    if let Some(git_dir) = git_dir {
        let tag_output = Command::new("git")
            .args(&["describe", "--tags", "--exact-match"])
            .output()
            .ok();

        let version = if let Some(tag_output) = tag_output {
            if tag_output.status.success() {
                String::from_utf8_lossy(&tag_output.stdout)
                    .trim()
                    .to_string()
            } else {
                let fallback_output = Command::new("git")
                    .args(&["describe", "--always", "--tags"])
                    .output()
                    .expect("Failed to run git describe");
                String::from_utf8_lossy(&fallback_output.stdout)
                    .trim()
                    .to_string()
            }
        } else {
            "unknown".to_string()
        };

        println!("cargo:rustc-env={}={}", var_name, version);

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let git_dir_path = PathBuf::from(&manifest_dir).join(&git_dir);
        println!("cargo:rerun-if-changed={}/HEAD", git_dir_path.display());
        println!(
            "cargo:rerun-if-changed={}/refs/heads",
            git_dir_path.display()
        );
        println!(
            "cargo:rerun-if-changed={}/refs/tags",
            git_dir_path.display()
        );
        return;
    }

    // Fallback if not in a git repository
    println!("cargo:rustc-env={}={}", var_name, "unknown");
}
