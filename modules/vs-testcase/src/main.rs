use clap::{error::ErrorKind, Parser, Subcommand};
use log::{error, info, LevelFilter};
use serde::Serialize;
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
        #[arg(short, long, default_value = "")]
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
        .filter_level(LevelFilter::Debug)
        .init();

    // make sure the program return non-zero status code when arguments are invalid
    let cli_args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => {
            // Check if the error is for displaying help or version
            match e.kind() {
                ErrorKind::DisplayVersion => {
                    println!(
                        "vesyla ({}) {}",
                        env!("CARGO_PKG_NAME"),
                        env!("VESYLA_VERSION")
                    );
                    process::exit(0);
                }
                ErrorKind::DisplayHelp => {
                    println!("{}", e);
                    process::exit(0);
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
    let testcase_dir = get_testcase_dir(None);
    copy_dir_all(&testcase_dir, &output_dir).unwrap();
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
    file.write_all(format!("{:?}", std::time::SystemTime::now()).as_bytes())
        .expect("Failed to write lock file");

    // get the current executable path
    let current_exe = env::current_exe().unwrap();
    let current_exe_dir = current_exe.parent().unwrap();
    let usr_dir = current_exe_dir.parent().unwrap();

    // construct template path
    let template_path = Path::new(usr_dir).join("share/vesyla/template").join(style);

    // check if template path exists
    if !Path::new(&template_path).exists() {
        error!("Template not found for style {}", style);
        process::exit(1);
    }

    // copy all the contents including files and subdirectories in template directory to the output directory
    copy_dir_all(&template_path, &output).expect("Failed to copy template directory");
}

fn run(directory: &String) {
    let test_dir = get_testcase_dir(Some(Path::new(directory).to_path_buf()));

    // check if the directory is the current directory, if yes, exit
    if test_dir == Path::new(".").canonicalize().unwrap() {
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
    let testcases_dir = get_testcase_dir(Some(Path::new(&directory).to_path_buf()));

    info!("Testcase directory: {:?}", testcases_dir);

    // Find all nth-level directories under the testcases directory
    fn collect_dirs_at_depth(root: &Path, depth: u8) -> Vec<String> {
        let mut result = Vec::new();
        let mut stack = vec![(root.to_path_buf(), 0)];

        while let Some((path, cur_depth)) = stack.pop() {
            if cur_depth == depth && path.is_dir() {
                result.push(path.to_str().unwrap().to_owned());
            } else if cur_depth < depth {
                if let Ok(entries) = fs::read_dir(&path) {
                    for entry in entries.flatten() {
                        let entry_path = entry.path();
                        if entry_path.is_dir() {
                            stack.push((entry_path, cur_depth + 1));
                        }
                    }
                }
            }
        }
        result
    }
    let leaf_path_vec: Vec<String> = collect_dirs_at_depth(Path::new(&testcases_dir), 3);

    #[derive(Serialize)]
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
    let testcase_path_str = get_testcase_dir(None).to_str().unwrap().to_string();
    for tc in &mut testcase_entries {
        if tc.path.starts_with(&testcase_path_str) {
            tc.path = tc.path.replace(&testcase_path_str, "");
        }
    }

    info!("Found {} testcases", testcase_entries.len());

    // generate the testcase scripts: run.sh
    let mut run_sh = File::create("run.sh").expect("Failed to create run.sh");
    let run_sh_content = include_str!("../assets/run.sh");
    run_sh
        .write_all(run_sh_content.as_bytes())
        .expect("Failed to write run.sh");

    // make the run.sh executable
    let mut perms = fs::metadata("run.sh").unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions("run.sh", perms).expect("Failed to set permissions");

    // generate the testcase scripts: autotest_config.robot
    let template = include_str!("../assets/autotest_template.robot.j2");
    let mut context = minijinja::Environment::new();
    context.add_template("autotest_template", template).unwrap();
    let result = context
        .get_template("autotest_template")
        .unwrap()
        .render(&testcase_entries);
    let comment =
        "*** Comments ***\nThis file was automatically generated by Vesyla. DO NOT EDIT.\n\n"
            .to_string();
    let output = comment + &result.expect("Failed to render template");
    let mut autotest_config_robot =
        File::create("autotest_config.robot").expect("Failed to create autotest_config.robot");
    autotest_config_robot
        .write_all(output.as_bytes())
        .expect("Failed to write autotest_config.robot");

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

fn get_testcase_dir(overwrite: Option<std::path::PathBuf>) -> std::path::PathBuf {
    let testcase_dir;
    if let Some(overwrite) = overwrite {
        testcase_dir = overwrite;
    } else {
        let current_exe = env::current_exe().unwrap();
        let current_exe_dir = current_exe.parent().unwrap();
        let usr_dir = current_exe_dir.parent().unwrap();
        testcase_dir = Path::new(usr_dir).join("share/vesyla/testcase");
    }

    // check if the directory exists
    if !Path::new(&testcase_dir).exists() {
        error!("Directory {:?} does not exist", &testcase_dir);
        process::exit(1);
    }
    // check if the directory is a directory
    if !Path::new(&testcase_dir).is_dir() {
        error!("{:?} is not a directory", &testcase_dir);
        process::exit(1);
    }

    return testcase_dir;
}
