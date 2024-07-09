use chrono::Local;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process;
use glob::glob;
use log::{info, warn, error, debug, trace};
use argparse;
use std::os::unix::fs::PermissionsExt;
use std::io;

fn main (){
    // set logger level to be debug
    env_logger::builder().filter_level(log::LevelFilter::Debug).init();

    // parse arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        error!("Usage: {} [command] [options]", args[0]);
        process::exit(1);
    }

    let command = &args[1];
    let mut options : Vec<String> = Vec::new();
    for i in 1..args.len() {
        options.push(args[i].clone());
    }
    
    match command.as_str() {
        "init" => {
            info!("Initializing ...");
            init(options)
        },
        "run" => {
            info!("Running testcase ...");
            run(options)
        },
        "generate" => {
            info!("Generating testcase scripts ...");
            generate(options)
        },
        _ => {
            error!("Unknown command: {}", command);
            panic!();
        }
    }
}

fn init(args: Vec<String>) {
    // parse the "args" using argparse
    // -s <style> -f -o <output directory>
    let mut style = String::from("drra");
    let mut force = false;
    let mut output = String::from(".");
    {
        let mut ap = argparse::ArgumentParser::new();
        ap.set_description("Initialize testcase directory");
        ap.refer(&mut style)
            .add_option(&["-s", "--style"], argparse::Store, "Template style");
        ap.refer(&mut force)
            .add_option(&["-f", "--force"], argparse::StoreTrue, "Force initialization");
        ap.refer(&mut output)
            .add_option(&["-o", "--output"], argparse::Store, "Output directory");
        ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr()).unwrap();
    }

    // create the output directory
    if !Path::new(&output).exists() {
        fs::create_dir_all(&output).unwrap();
    }

    // lock the output directory
    let lock_file = format!("{}/.lock", output);
    if Path::new(&lock_file).exists() {
        if force {
            fs::remove_file(&lock_file).expect("Failed to remove lock file");
        } else {
            error!("Directory is already initialized. Use -f to force re-initialization");
            process::exit(1);
        }
    }

    // create the lock file and write the current timestamp
    let mut file = File::create(&lock_file).expect("Failed to create lock file");
    file.write_all(format!("{}", Local::now()).as_bytes()).expect("Failed to write lock file");

    // get VESYLA_SUITE_PATH_PROG environment variable
    let prog_path = env::var("VESYLA_SUITE_PATH_PROG").expect("VESYLA_SUITE_PATH_PROG not set");

    // construct template path
    let template_path = format!("{}/share/vesyla-suite/template/{}", prog_path, style);

    // copy all the contents including files and subdirectories in template directory to the output directory
    copy_dir_all(&template_path, &output).expect("Failed to copy template directory");
}

fn run(args: Vec<String>) {
    assert! (args.len() ==  2);
    let test_dir = &args[1];

    // initialize the testcase directory
    init(vec!["init", "-f", "-s", "drra", "-o", "."].iter().map(|s| s.to_string()).collect());

    // copy everything from the test directory to the current directory
    copy_dir_all(&test_dir, ".").unwrap();

    // run the testcase
    process::Command::new("sh")
        .arg("run.sh")
        .output()
        .expect("Failed to run testcase");
}

fn generate(args: Vec<String>){
    assert! (args.len() ==  2);
    let testcases_dir = &args[1];

    // Find all third-level directories under the testcases directory
    let mut leaf_path_vec: Vec<String> = Vec::new();
    let pattern = Path::new(testcases_dir).join("*").join("*").join("*");
    let pattern = pattern.to_str().expect("Failed to convert path to string");
    for entry in glob(pattern).expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                if path.is_dir() {
                    // convert the path to string
                    let leaf_path_str = path.to_str().expect("Path is not valid UTF-8").to_owned();
                    leaf_path_vec.push(leaf_path_str);
                }
            },
            Err(e) => {
                error!("Error: {:?}", e);
            }
        }
    }

    struct TestcaseEntry {
        name: String,
        tags: String,
        path: String
    }

    let mut testcase_entries = Vec::new();
    for leaf_path in leaf_path_vec {
        let leaf_path_str = leaf_path;
        let leaf_path_str_split: Vec<&str> = leaf_path_str.split("/").collect();
        let name = leaf_path_str_split[(leaf_path_str_split.len()-3)..].join("::");
        let tags = leaf_path_str_split[(leaf_path_str_split.len()-3)..(leaf_path_str_split.len()-1)].join("::");
        let path = leaf_path_str;
        testcase_entries.push(TestcaseEntry{name: name, tags: tags, path: path});
    }

    info!("Found {} testcases", testcase_entries.len());

    // generate the testcase scripts: run.sh
    let mut run_sh = File::create("run.sh").expect("Failed to create run.sh");
    run_sh.write_all(r##"
#!/bin/sh
pabot --testlevelsplit -d output autotest_config.robot
"##.as_bytes()).expect("Failed to write run.sh");

    // make the run.sh executable
    let mut perms = fs::metadata("run.sh").unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions("run.sh", perms).expect("Failed to set permissions");

    // generate the testcase scripts: autotest_config.robot
    let mut autotest_config_robot = File::create("autotest_config.robot").expect("Failed to create autotest_config.robot");
    autotest_config_robot.write_all(r#"
*** Settings ***
Library           Process
Library           OperatingSystem
Library           String
Suite Teardown    Terminate All Processes    kill=True
Test Template     Autotest Template

*** Test Cases ***  filename
"#.as_bytes()).expect("Failed to write autotest_config.robot");
        for tc in testcase_entries {
            autotest_config_robot.write(format!("tc {}    {}\n    [Tags]    {}\n", tc.name, tc.path, tc.tags).as_bytes()).expect("Failed to write autotest_config.robot");
        }

        autotest_config_robot.write_all(r#"
*** Keywords ***
Autotest Template
    [Arguments]  ${filename}
    ${random_string} =    Generate Random String    12    [LOWER]
    Create Directory    work/${random_string}
    ${result} =    Run Process    vs-testcase run ${filename}    shell=True    timeout=30 min    stdout=stdout.txt    stderr=stderr.txt    cwd=work/${random_string}
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