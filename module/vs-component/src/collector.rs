pub fn collect_rtl(
    arch_file: &String,
    output_dir: &String,
    component_map: &std::collections::HashMap<String, String>,
) {
    // read the json file
    let json_str = std::fs::read_to_string(arch_file).expect("Failed to read file");
    let arch: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

    // create the output directory if it does not exist
    std::fs::create_dir_all(&output_dir).unwrap();

    // create directories for the rtl files
    let rtl_dir = format!("{}/rtl", output_dir);
    std::fs::create_dir_all(&rtl_dir).unwrap();

    // for each template file, copy it to the output directory and render it by jinja2 with the arch json, then write the result to the output file. Keep the directory structure.
    let vesyla_suite_path_template = std::env::var("VESYLA_SUITE_PATH_TEMPLATE").expect("Environment variable VESYLA_SUITE_PATH_TEMPLATE not set! Did you forget to source the setup script env.sh?");
    let fabric_template_dir = format!("{}/drra_rtl_fabric", vesyla_suite_path_template);
    let env = minijinja::Environment::new();
    // create a enclosure to process a directory
    fn process_dir(
        dir: &str,
        rtl_dir: &str,
        arch: &serde_json::Value,
        env: &minijinja::Environment,
    ) {
        let files = std::fs::read_dir(&dir).unwrap();
        for file in files {
            // if the file is a directory, process it recursively
            if file.as_ref().unwrap().path().is_dir() {
                process_dir(file.unwrap().path().to_str().unwrap(), rtl_dir, arch, env);
                continue;
            } else {
                let file = file.unwrap();
                let file_path = file.path();
                let file_name = file.file_name();
                let file_name = file_name.to_str().unwrap();
                // check the file extension, if extension is ".j2" or ".jinja2", render it. Drop the extension and copy it to the output directory.
                if !file_name.ends_with(".j2") && !file_name.ends_with(".jinja2") {
                    // copy the file to the output directory
                    let output_file = format!("{}/{}", rtl_dir, file_name);
                    std::fs::copy(&file_path, &output_file).unwrap();
                } else {
                    // change filename and remove the file extension
                    let file_name = file_name.replace(".j2", "").replace(".jinja2", "");
                    let output_file = format!("{}/{}", rtl_dir, file_name);
                    let tmpl = env.get_template(&file_name).unwrap();
                    let result = tmpl.render(minijinja::context!(arch)).unwrap();
                    std::fs::write(&output_file, result).expect("Failed to write file");
                }
            }
        }
    }
    process_dir(&fabric_template_dir, &rtl_dir, &arch, &env);

    // create the components directory if not exist and copy the components (api.sv and rtl.sv) in the arch json to the directory.
    let components_dir = format!("{}/components", output_dir);
    std::fs::create_dir_all(&components_dir).unwrap();
    let cell_dir = format!("{}/cell", components_dir);
    std::fs::create_dir_all(&cell_dir).unwrap();
    let controller_dir = format!("{}/controller", components_dir);
    std::fs::create_dir_all(&controller_dir).unwrap();
    let resource_dir = format!("{}/resource", components_dir);
    std::fs::create_dir_all(&resource_dir).unwrap();
    // copy cells
    let cells = arch["cells"].as_array().unwrap();
    for cell in cells {
        let cell_name = cell["name"].as_str().unwrap();
        let src_file = format!("{}/rtl.sv", component_map[cell_name]);
        let dst_file = format!("{}/{}/rtl.sv", cell_dir, cell_name);
        std::fs::copy(&src_file, &dst_file).unwrap();
    }
    // copy controllers
    let controllers = arch["controllers"].as_array().unwrap();
    for controller in controllers {
        let controller_name = controller["name"].as_str().unwrap();
        let src_file = format!("{}/rtl.sv", component_map[controller_name]);
        let dst_file = format!("{}/{}/rtl.sv", controller_dir, controller_name);
        std::fs::copy(&src_file, &dst_file).unwrap();
        let src_file = format!("{}/api.sv", component_map[controller_name]);
        let dst_file = format!("{}/{}/api.sv", controller_dir, controller_name);
        std::fs::copy(&src_file, &dst_file).unwrap();
    }
    // copy resources
    let resources = arch["resources"].as_array().unwrap();
    for resource in resources {
        let resource_name = resource["name"].as_str().unwrap();
        let src_file = format!("{}/rtl.sv", component_map[resource_name]);
        let dst_file = format!("{}/{}/rtl.sv", resource_dir, resource_name);
        std::fs::copy(&src_file, &dst_file).unwrap();
        let src_file = format!("{}/api.sv", component_map[resource_name]);
        let dst_file = format!("{}/{}/api.sv", resource_dir, resource_name);
        std::fs::copy(&src_file, &dst_file).unwrap();
    }
}
