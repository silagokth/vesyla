use log::error;
use std::env;
use std::process;

fn get_drra_components_version() -> Result<String, std::io::Error> {
    // Check if the VESYLA_SUITE_PATH_COMPONENTS environment variable is set
    if env::var("VESYLA_SUITE_PATH_COMPONENTS").is_err() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "VESYLA_SUITE_PATH_COMPONENTS environment variable is not set",
        ));
    }

    // Read the version file (library/VERSION)
    let drra_components_path = env::var("VESYLA_SUITE_PATH_COMPONENTS").unwrap();
    let version_file_path = format!("{}/VERSION", drra_components_path);
    let version_content = std::fs::read_to_string(version_file_path)?;

    Ok(version_content.trim().to_string())
}

fn main() {
    // define the used environment variables
    let name_list = vec!["VESYLA_SUITE_PATH_TESTCASE".to_string()];

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
         \tcompile     Compile the source code\n\
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
    let tools_list = vec!["compile", "component", "manas", "schedule", "testcase"];
    match command.as_str() {
        "-h" | "--help" => {
            println!("{}", help_message);
            process::exit(0);
        }
        "-V" | "--version" => {
            println!("vesyla {}", env!("VESYLA_VERSION"));
            match get_drra_components_version() {
                Ok(version) => {
                    println!("drra-components {}", version);
                }
                Err(err) => {
                    println!("drra-components version unknown");
                    error!("Failed to retrieve drra-components version: {}", err);
                }
            }
        }
        cmd if tools_list.contains(&cmd) => {
            let current_exe = env::current_exe().unwrap();
            let current_exe_dir = current_exe.parent().unwrap();
            let prog_path = current_exe_dir.join(format!("vs-{}", command));
            let prog = prog_path.to_str().unwrap();

            let status = process::Command::new(prog)
                .args(&args[2..])
                .stdout(process::Stdio::inherit())
                .stderr(process::Stdio::inherit())
                .status()
                .unwrap_or_else(|_| panic!("Failed to execute command: vs-{}", command));
            if !status.success() && status.code() != Some(2) {
                error!("{} command failed", command);
                error!("Exit code: {}", status.code().unwrap_or(-1));
                process::exit(status.code().unwrap_or(-1));
            }
        }
        _ => {
            error!("Unknown command: {}", command);
            process::exit(1);
        }
    }

    // restore the environment variables
    pop_env(&name_list, saved_env);
}

fn init() {
    // set environment variable
    let current_exe = env::current_exe().unwrap();
    let current_exe_dir = current_exe.parent().unwrap();
    let vesyla_suite_path_testcase = current_exe_dir
        .parent()
        .unwrap()
        .join("share/vesyla/testcase");

    unsafe {
        env::set_var("VESYLA_SUITE_PATH_TESTCASE", vesyla_suite_path_testcase);
    }
}

fn push_env(name_list: &Vec<String>) -> Vec<(String, Option<String>)> {
    let mut saved_env: Vec<(String, Option<String>)> = Vec::new();
    for name in name_list {
        match env::var(name) {
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
            Some(val) => unsafe {
                env::set_var(&name, val);
            },
            None => unsafe {
                env::remove_var(&name);
            },
        }
    }
}
