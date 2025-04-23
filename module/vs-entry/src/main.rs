use log::{error, info, warn};
use regex;
use std::env;
use std::fs;
use std::process;

fn main() {
    // define the used environment variables
    let name_list = vec![
        "VESYLA_SUITE_PATH_PROG".to_string(),
        "VESYLA_SUITE_PATH_BIN".to_string(),
        "VESYLA_SUITE_PATH_LIB".to_string(),
        "VESYLA_SUITE_PATH_INC".to_string(),
        "VESYLA_SUITE_PATH_SHARE".to_string(),
        "VESYLA_SUITE_PATH_TEMPLATE".to_string(),
        "VESYLA_SUITE_PATH_TESTCASE".to_string(),
        "VESYLA_SUITE_PATH_TMP".to_string(),
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION".to_string(),
    ];

    // save the environment variables
    let saved_env = push_env(&name_list);

    // initialize the environment variables
    init();

    // set logger level to be debug
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();
    log_panics::init();

    let args: Vec<String> = env::args().collect();

    let help_message = format!(
        "Usage: {} [command and options]\n\
         Commands:\n\
         \tcomponent   Assemble the system\n\
         \tmanas       Validate JSON file\n\
         \tschedule    Clean the build directory\n\
         \ttestcase    Test the system\n\
         Options:\n\
         \t-h, --help     Show this help message\n\
         \t-V, --version  Show version information",
        args[0]
    );

    if args.len() < 2 {
        error!("{}", help_message);
        process::exit(1);
    }

    // find the directory of the current executable
    let command = &args[1];
    let tools_list = vec!["component", "manas", "schedule", "testcase"];
    match command.as_str() {
        "-h" | "--help" => {
            info!("{}", help_message);
            process::exit(0);
        }
        "-V" | "--version" => {
            let version = extract_version();
            let mut tools_versions = vec![];
            for tool in tools_list.iter() {
                let prog = env::var("VESYLA_SUITE_PATH_BIN").unwrap().to_string() + "/vs-" + tool;
                let status = process::Command::new(prog)
                    .arg("--version")
                    .stdout(process::Stdio::piped())
                    .stderr(process::Stdio::inherit())
                    .output()
                    .expect("failed to execute process");
                if status.status.success() {
                    let output = String::from_utf8_lossy(&status.stdout);
                    let tool_version = output
                        .trim()
                        .split_whitespace()
                        .last()
                        .unwrap_or("Unknown")
                        .to_string();
                    tools_versions.push(format!("{}: {}", tool, tool_version));
                } else {
                    tools_versions.push(format!("{}: Unknown", tool));
                    warn!("{:?}", status.stderr);
                    warn!("Failed to get version for tool: {}", tool);
                }
            }
            info!("vesyla-suite {}", version);
            for tool_version in tools_versions {
                info!(" -> {}", tool_version);
            }
        }
        cmd if tools_list.contains(&cmd) => {
            let prog = env::var("VESYLA_SUITE_PATH_BIN").unwrap().to_string() + "/vs-" + command;
            let status = process::Command::new(prog)
                .args(&args[2..])
                .stdout(process::Stdio::inherit())
                .stderr(process::Stdio::inherit())
                .status()
                .expect("failed to execute process");
            if !status.success() {
                if status.code() != Some(2) {
                    error!("{} command failed", command);
                    info!("Exit code: {}", status.code().unwrap_or(-1));
                    process::exit(status.code().unwrap_or(-1));
                }
            }
        }
        _ => {
            error!("Unknown command: {}", command);
            process::exit(1);
        }
    }

    // finalize the environment variables
    finish();

    // restore the environment variables
    pop_env(&name_list, saved_env);
}

fn extract_version() -> String {
    let changelog = include_str!("../../../CHANGELOG.md");
    let mut version = "Unknown".to_string();
    for line in changelog.lines() {
        if line.starts_with("## ") || line.starts_with("##\t") {
            let title = line[3..].to_string().trim().to_string();
            // if it matches the pattern "## [x.y.z]..."
            if let Some(captures) = regex::Regex::new(r"^\[([0-9]+\.[0-9]+\.[0-9]+)\].*$")
                .unwrap()
                .captures(&title)
            {
                version = captures[1].to_string();
                break;
            }
        }
    }
    version
}

fn init() {
    // set environment variable
    let current_exe = env::current_exe().unwrap();
    let current_exe_dir = current_exe.parent().unwrap();
    let vesyla_suite_path_prog = current_exe_dir.parent().unwrap();
    let vesyla_suite_path_bin = vesyla_suite_path_prog.join("bin");
    let vesyla_suite_path_lib = vesyla_suite_path_prog.join("lib");
    let vesyla_suite_path_inc = vesyla_suite_path_prog.join("include");
    let vesyla_suite_path_share = vesyla_suite_path_prog.join("share/vesyla-suite");
    let vesyla_suite_path_template = vesyla_suite_path_share.join("template");
    let vesyla_suite_path_testcase = vesyla_suite_path_share.join("testcase");
    let random_number: u32 = rand::random();
    let vesyla_suite_path_tmp = format!("/tmp/vesyla_suite_{}", random_number);

    env::set_var("VESYLA_SUITE_PATH_PROG", vesyla_suite_path_prog);
    env::set_var("VESYLA_SUITE_PATH_BIN", vesyla_suite_path_bin);
    env::set_var("VESYLA_SUITE_PATH_LIB", vesyla_suite_path_lib);
    env::set_var("VESYLA_SUITE_PATH_INC", vesyla_suite_path_inc);
    env::set_var("VESYLA_SUITE_PATH_SHARE", vesyla_suite_path_share);
    env::set_var("VESYLA_SUITE_PATH_TEMPLATE", vesyla_suite_path_template);
    env::set_var("VESYLA_SUITE_PATH_TESTCASE", vesyla_suite_path_testcase);
    env::set_var("VESYLA_SUITE_PATH_TMP", vesyla_suite_path_tmp);

    // set protobuf for python
    env::set_var("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python");

    // create the temporary directory
    let path = env::var("VESYLA_SUITE_PATH_TMP").unwrap();
    fs::create_dir_all(path).unwrap();
}

fn finish() {
    // remove the temporary directory
    let path = env::var("VESYLA_SUITE_PATH_TMP").unwrap();
    fs::remove_dir_all(path).unwrap();
}

fn push_env(name_list: &Vec<String>) -> Vec<(String, Option<String>)> {
    let mut saved_env: Vec<(String, Option<String>)> = Vec::new();
    for name in name_list {
        match env::var(&name) {
            Ok(val) => {
                saved_env.push((name.to_string(), Some(val)));
            }
            Err(_) => {
                saved_env.push((name.to_string(), None));
            }
        }
    }
    saved_env
}

fn pop_env(name_list: &Vec<String>, saved_env: Vec<(String, Option<String>)>) {
    let _ = name_list;
    for (name, val) in saved_env {
        match val {
            Some(val) => {
                env::set_var(name.to_string(), val);
            }
            None => {
                env::remove_var(name.to_string());
            }
        }
    }
}
