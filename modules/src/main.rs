use log::{error, info};
use std::env;
use std::process;

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
            println!("vesyla {}", env!("VESYLA_VERSION"));
        }
        cmd if tools_list.contains(&cmd) => {
            let current_exe = env::current_exe().unwrap();
            let current_exe_dir = current_exe.parent().unwrap();
            let prog_path = current_exe_dir.join(format!("vs-{}", command));
            let prog = prog_path.to_str().unwrap();

            let status = process::Command::new(&prog)
                .args(&args[2..])
                .stdout(process::Stdio::inherit())
                .stderr(process::Stdio::inherit())
                .status()
                .expect(format!("Failed to execute command: vs-{}", command).as_str());
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
            Some(val) => unsafe {
                env::set_var(name.to_string(), val);
            },
            None => unsafe {
                env::remove_var(name.to_string());
            },
        }
    }
}
