use chrono::Local;
use clap::{error::ErrorKind, Parser, Subcommand};
use glob::glob;
use log::{error, info};
use std::env;
use std::fs::{self, File};
use std::io;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process;

#[derive(Subcommand)]
enum Command {
    #[command(about = "Initialize testcase directory", name = "init")]
    Init {
        /// Template style
        #[arg(short, long, default_value = "drra")]
        style: String,
        /// Force initialization
        #[arg(short, long, default_value = "false")]
        force: bool,
        /// Output directory
        #[arg(short, long, default_value = ".")]
        output: String,
    },
    #[command(about = "Run testcase", name = "run")]
    Run {
        /// Testcase directory
        #[arg(short, long)]
        directory: String,
    },
    #[command(about = "Generate testcase scripts", name = "generate")]
    Generate {
        /// Testcase directory
        #[arg(short, long, default_value = "{{VESYLA_SUITE_PATH_TESTCASE}}")]
        directory: String,
    },
    #[command(about = "Export testcase", name = "export")]
    Export {
        /// Output directory
        #[arg(short, long, default_value = ".")]
        output: String,
    },
}

#[derive(Parser)]
#[command(version, about, long_about = None, allow_missing_positional = true, after_help = "")]
struct Args {
    /// Command to execute
    #[command(subcommand)]
    command: Command,
}

fn main() {
    // set logger level to be debug
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();
    log_panics::init();

    // make sure the program return non-zero status code when arguments are invalid
    let cli_args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => {
            // Check if the error is for displaying help or version
            match e.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                    // clap already printed the info
                    e.print().expect("Failed to print clap help/version");
                    process::exit(0); // Exit successfully
                }
                // For any other parsing error
                _ => {
                    // Log the error message provided by clap
                    error!("{}", e);
                    process::exit(1); // Exit with an error code
                }
            }
        }
    };

    match &cli_args.command {
        Command::Init {
            style,
            force,
            output,
        } => {
            info!("Initializing ...");
            init(style, force, output);
        }
        Command::Run { directory } => {
            info!("Running testcase ...");
            run(directory)
        }
        Command::Generate { directory } => {
            info!("Generating testcase scripts ...");
            generate(directory)
        }
        Command::Export { output } => {
            info!("Exporting testcase ...");
            export(output)
        }
    }
}

fn export(output: &String) {
    // convert the output path to absolute path
    let output_dir = output.clone();

    // check if the output directory exists, if not create it
    if !Path::new(&output_dir).exists() {
        fs::create_dir_all(&output_dir).unwrap();
    }

    // copy everything from default test directory to the output directory
    let vesyla_suite_path_testcase =
        env::var("VESYLA_SUITE_PATH_TESTCASE").expect("VESYLA_SUITE_PATH_TESTCASE not set");
    copy_dir_all(&vesyla_suite_path_testcase, &output_dir).unwrap();
}

fn init(style: &String, force: &bool, output: &String) {
    // create the output directory
    if !Path::new(&output).exists() {
        fs::create_dir_all(&output).unwrap();
    }

    // lock the output directory
    let lock_file = format!("{}/.lock", output);
    if Path::new(&lock_file).exists() {
        if *force {
            fs::remove_file(&lock_file).expect("Failed to remove lock file");
        } else {
            error!("Directory is already initialized. Use -f to force re-initialization");
            process::exit(1);
        }
    }

    // create the lock file and write the current timestamp
    let mut file = File::create(&lock_file).expect("Failed to create lock file");
    file.write_all(format!("{}", Local::now()).as_bytes())
        .expect("Failed to write lock file");

    // get VESYLA_SUITE_PATH_PROG environment variable
    let prog_path = env::var("VESYLA_SUITE_PATH_PROG").expect("VESYLA_SUITE_PATH_PROG not set");

    // construct template path
    let template_path = format!("{}/share/vesyla-suite/template/{}", prog_path, style);

    // check if template path exists
    if !Path::new(&template_path).exists() {
        error!("Template not found for style {}", style);
        process::exit(1);
    }

    // copy all the contents including files and subdirectories in template directory to the output directory
    copy_dir_all(&template_path, &output).expect("Failed to copy template directory");
}

fn run(directory: &String) {
    // if test_dir starts with {{VESYLA_SUITE_PATH_TESTCASE}}, replace it with the actual path
    let vesyla_suite_path_testcase =
        env::var("VESYLA_SUITE_PATH_TESTCASE").expect("VESYLA_SUITE_PATH_TESTCASE not set");
    let test_dir = directory.replace(
        "{{VESYLA_SUITE_PATH_TESTCASE}}",
        &vesyla_suite_path_testcase,
    );

    // check if the directory exists
    if !Path::new(&test_dir).exists() {
        error!("Directory {} does not exist", &test_dir);
        process::exit(1);
    }
    // check if the directory is a directory
    if !Path::new(&test_dir).is_dir() {
        error!("{} is not a directory", &test_dir);
        process::exit(1);
    }

    // convert the directory path to absolute path
    let test_dir = Path::new(&test_dir)
        .canonicalize()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    // check if the directory is the current directory, if yes, exit
    if test_dir == Path::new(".").canonicalize().unwrap().to_str().unwrap() {
        error!("Testcase source directory cannot be the current working directory!");
        process::exit(1);
    }

    // use init() to initialize the testcase directory
    init(&"drra".to_string(), &true, &".".to_string());

    // copy everything from the test directory to the current directory
    copy_dir_all(&test_dir, ".").unwrap();

    // run the testcase, if the testcase fails, the process will exit with non-zero status
    let status = process::Command::new("sh")
        .arg("run.sh")
        .status()
        .expect("Failed to run the testcase");
    assert!(status.success());
}

fn generate(directory: &String) {
    // replace {{VESYLA_SUITE_PATH_TESTCASE}} with the actual path
    let vesyla_suite_path_testcase =
        env::var("VESYLA_SUITE_PATH_TESTCASE").expect("VESYLA_SUITE_PATH_TESTCASE not set");
    let real_dir = directory.replace(
        "{{VESYLA_SUITE_PATH_TESTCASE}}",
        &vesyla_suite_path_testcase,
    );

    // check if the directory exists
    if !Path::new(&real_dir).exists() {
        error!("Directory {} does not exist", &real_dir);
        process::exit(1);
    }
    // check if the directory is a directory
    if !Path::new(&real_dir).is_dir() {
        error!("{} is not a directory", &real_dir);
        process::exit(1);
    }

    // convert the testcases directory path to absolute path
    let testcases_dir = Path::new(&real_dir)
        .canonicalize()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    info!("Testcase directory: {}", testcases_dir);

    // Find all third-level directories under the testcases directory
    let mut leaf_path_vec: Vec<String> = Vec::new();
    let pattern = Path::new(&testcases_dir).join("*").join("*").join("*");
    let pattern = pattern.to_str().expect("Failed to convert path to string");
    for entry in glob(pattern).expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                if path.is_dir() {
                    // convert the path to string
                    let leaf_path_str = path.to_str().expect("Path is not valid UTF-8").to_owned();
                    leaf_path_vec.push(leaf_path_str);
                }
            }
            Err(e) => {
                error!("Error: {:?}", e);
            }
        }
    }

    struct TestcaseEntry {
        name: String,
        tags: String,
        path: String,
    }

    let mut testcase_entries = Vec::new();
    for leaf_path in leaf_path_vec {
        let leaf_path_str = leaf_path;
        let leaf_path_str_split: Vec<&str> = leaf_path_str.split("/").collect();
        let name = leaf_path_str_split[(leaf_path_str_split.len() - 3)..].join("::");
        let tags = leaf_path_str_split
            [(leaf_path_str_split.len() - 3)..(leaf_path_str_split.len() - 1)]
            .join("::");
        let path = leaf_path_str;
        // if path is relative path, convert it to absolute path using canonicalize
        let path = Path::new(&path)
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        testcase_entries.push(TestcaseEntry {
            name: name,
            tags: tags,
            path: path,
        });
    }

    // if path is under VESYLA_SUITE_PATH_TESTCASE, convert it to relative path by replace it with {{VESYLA_SUITE_PATH_TESTCASE}}
    for tc in &mut testcase_entries {
        if tc.path.starts_with(&vesyla_suite_path_testcase) {
            tc.path = tc.path.replace(
                &vesyla_suite_path_testcase,
                "{{VESYLA_SUITE_PATH_TESTCASE}}",
            );
        }
    }

    info!("Found {} testcases", testcase_entries.len());

    // generate the testcase scripts: run.sh
    let mut run_sh = File::create("run.sh").expect("Failed to create run.sh");
    run_sh
        .write_all(
            r##"
#!/bin/sh
set -e
pabot --testlevelsplit -d output autotest_config.robot
"##
            .as_bytes(),
        )
        .expect("Failed to write run.sh");

    // make the run.sh executable
    let mut perms = fs::metadata("run.sh").unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions("run.sh", perms).expect("Failed to set permissions");

    // generate the testcase scripts: autotest_config.robot
    let mut autotest_config_robot =
        File::create("autotest_config.robot").expect("Failed to create autotest_config.robot");
    autotest_config_robot
        .write_all(
            r#"
*** Settings ***
Library           Process
Library           OperatingSystem
Library           String
Suite Teardown    Terminate All Processes    kill=True
Test Template     Autotest Template

*** Test Cases ***  filename
"#
            .as_bytes(),
        )
        .expect("Failed to write autotest_config.robot");
    for tc in testcase_entries {
        autotest_config_robot
            .write(format!("tc {}    {}\n    [Tags]    {}\n", tc.name, tc.path, tc.tags).as_bytes())
            .expect("Failed to write autotest_config.robot");
    }

    autotest_config_robot.write_all(r#"
*** Keywords ***
Autotest Template
    [Arguments]  ${filename}
    ${random_string} =    Generate Random String    12    [LOWER]
    Create Directory    work/${random_string}
    ${result} =    Run Process    vesyla-suite testcase run -d "${filename}"    shell=True    timeout=30 min    stdout=stdout.txt    stderr=stderr.txt    cwd=work/${random_string}
    Should Be Equal As Integers    ${result.rc}    0
    Remove Directory   work/${random_string}     recursive=True
"#.as_bytes()).expect("Failed to write autotest_config.robot");

    // create the work directory
    fs::create_dir_all("work").unwrap();
}

fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
    fs::create_dir_all(&dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
        }
    }
    Ok(())
}
