use crate::drra::Fabric;
use crate::utils;
use std::collections::HashMap;
use std::fs;
use svg::node::element::{Group, Line, Rectangle, Text};
use svg::Document;

pub fn generate(arch_file: &String, output_dir: &String) {
    gen_picture(arch_file, output_dir);
}

fn gen_picture(arch_file: &String, output_dir: &String) {
    // if the output directory does not exist, create it
    let output_dir_path = std::path::Path::new(output_dir);
    if !output_dir_path.exists() {
        fs::create_dir_all(output_dir_path).expect("Failed to create output directory");
    }

    // read the json file
    let json_str = std::fs::read_to_string(arch_file).expect("Failed to read file");
    let arch: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");
    // let fabric_json = arch.get("fabric").expect("Fabric not found in .json");
    let mut fabric = Fabric::new();
    fabric.add_parameters(utils::get_parameters(&arch, None));
    let _row: i64 = fabric.get_parameter("ROWS").unwrap().try_into().unwrap();
    let col: i64 = fabric.get_parameter("COLS").unwrap().try_into().unwrap();

    let mut color_map: HashMap<String, String> = HashMap::new();
    color_map.insert("buffer_color_fill".to_string(), "#B2E0E0".to_string());
    color_map.insert("buffer_color_border".to_string(), "#1C434C".to_string());
    color_map.insert("buffer_color_text".to_string(), "#1C434C".to_string());
    color_map.insert("cell_color_fill".to_string(), "#E6E6E6".to_string());
    color_map.insert("cell_color_border".to_string(), "#A5A5A5".to_string());
    color_map.insert("cell_color_text".to_string(), "#323232".to_string());
    color_map.insert("controller_color_fill".to_string(), "#FFF0B0".to_string());
    color_map.insert("controller_color_border".to_string(), "#A65900".to_string());
    color_map.insert("controller_color_text".to_string(), "#A65900".to_string());
    color_map.insert("resource_color_fill".to_string(), "#C7EBBA".to_string());
    color_map.insert("resource_color_border".to_string(), "#0D4A21".to_string());
    color_map.insert("resource_color_text".to_string(), "#0D4A21".to_string());

    let mut geometry_map: HashMap<String, i64> = HashMap::new();
    geometry_map.insert("controller_width".to_string(), 80);
    geometry_map.insert("controller_height".to_string(), 320);
    geometry_map.insert("resource_width".to_string(), 260);
    geometry_map.insert("resource_height".to_string(), 20);
    geometry_map.insert("cell_width".to_string(), 400);
    geometry_map.insert("cell_height".to_string(), 400);
    geometry_map.insert("buffer_height".to_string(), 100);
    geometry_map.insert("buffer_width".to_string(), col * geometry_map["cell_width"]);

    let doc = draw_fabric(arch, &color_map, &geometry_map);
    save_doc_to_files(doc, output_dir);
}

fn save_doc_to_files(doc: Document, output_dir: &String) {
    // write to file as svg
    let file_name = format!("{}/arch.svg", output_dir);
    svg::save(file_name, &doc).unwrap();
}

fn draw_fabric(
    j: serde_json::Value,
    color_map: &HashMap<String, String>,
    geometry_map: &HashMap<String, i64>,
) -> Document {
    // get row and col from fabric
    let mut fabric = Fabric::new();
    fabric.add_parameters(utils::get_parameters(&j, None));
    let row: i64 = fabric.get_parameter("ROWS").unwrap().try_into().unwrap();
    let col: i64 = fabric.get_parameter("COLS").unwrap().try_into().unwrap();

    // create svg document
    let mut document: Document = Document::new();

    // draw fabric
    let fabric_width = col * geometry_map["cell_width"];
    let fabric_height = row * geometry_map["cell_height"] + 2 * geometry_map["buffer_height"];
    document = document
        .set("width", fabric_width)
        .set("height", fabric_height)
        .set("viewBox", (0, 0, fabric_width, fabric_height));

    // draw buffers
    let offset = 5;
    let buffer_height = geometry_map["buffer_height"];
    let buffer_width = geometry_map["buffer_width"];
    let buffer_color_fill = color_map["buffer_color_fill"].clone();
    let buffer_color_border = color_map["buffer_color_border"].clone();
    let buffer_color_text = color_map["buffer_color_text"].clone();
    document = document
        .add(
            Rectangle::new()
                .set("x", offset)
                .set("y", offset)
                .set("width", buffer_width - 2 * offset)
                .set("height", buffer_height - 2 * offset)
                .set("fill", buffer_color_fill.clone())
                .set("stroke", buffer_color_border.clone())
                .set("stroke-width", 2),
        )
        .add(
            Text::new("Input Buffer")
                .set("x", buffer_width / 2)
                .set("y", buffer_height / 2)
                .set("fill", buffer_color_text.clone())
                .set("font-size", 20)
                .set("text-anchor", "middle"),
        )
        .add(
            Rectangle::new()
                .set("x", offset)
                .set(
                    "y",
                    offset + buffer_height + row * geometry_map["cell_height"],
                )
                .set("width", buffer_width - 2 * offset)
                .set("height", buffer_height - 2 * offset)
                .set("fill", buffer_color_fill.clone())
                .set("stroke", buffer_color_border.clone())
                .set("stroke-width", 2),
        )
        .add(
            Text::new("Output Buffer")
                .set("x", buffer_width / 2)
                .set(
                    "y",
                    offset + buffer_height + row * geometry_map["cell_height"] + buffer_height / 2,
                )
                .set("fill", buffer_color_text)
                .set("font-size", 20)
                .set("text-anchor", "middle"),
        );

    // draw cells
    let cell_width = geometry_map["cell_width"];
    let cell_height = geometry_map["cell_height"];
    let cell_color_fill = color_map["cell_color_fill"].clone();
    let cell_color_border = color_map["cell_color_border"].clone();
    let cell_color_text = color_map["cell_color_text"].clone();
    for c in j["cells"].as_array().unwrap() {
        let rr = c["coordinates"]["row"].as_i64().unwrap();
        let cc = c["coordinates"]["col"].as_i64().unwrap();
        let cell = c["cell"].as_object().unwrap();
        let controller = cell["controller"].as_object().unwrap();
        let resources = cell["resources_list"].as_array().unwrap();
        let x = cc * cell_width;
        let y = rr * cell_height + buffer_height;
        let mut cell_group = Group::new();

        cell_group = cell_group.add(
            Rectangle::new()
                .set("x", x + offset)
                .set("y", y + offset)
                .set("width", cell_width - 2 * offset)
                .set("height", cell_height - 2 * offset)
                .set("fill", cell_color_fill.clone())
                .set("stroke", cell_color_border.clone())
                .set("stroke-width", 2),
        );
        cell_group = cell_group.add(
            Text::new(format!("[{},{}]", rr, cc))
                .set("x", x + cell_width / 2)
                .set("y", y + 40)
                .set("fill", cell_color_text.clone())
                .set("font-size", 20)
                .set("font-weight", "bold")
                .set("text-anchor", "middle")
                .set("font-style", "normal"),
        );

        let controller_width = geometry_map["controller_width"];
        let controller_height = geometry_map["controller_height"];
        let controller_color_fill = color_map["controller_color_fill"].clone();
        let controller_color_border = color_map["controller_color_border"].clone();
        let controller_color_text = color_map["controller_color_text"].clone();
        cell_group = cell_group.add(
            Rectangle::new()
                .set("x", x + offset + 10 + 11)
                .set("y", y + offset + 50 + 11)
                .set("width", controller_width - 2)
                .set("height", controller_height - 2)
                .set("fill", controller_color_fill.clone())
                .set("stroke", controller_color_border.clone())
                .set("stroke-width", 2),
        );
        // add text rotate 90 degree
        let text_x = x + 11 + controller_width / 2;
        let text_y = y + 11 + controller_height / 2;
        cell_group = cell_group.add(
            Text::new(controller["name"].as_str().unwrap())
                .set("x", text_x)
                .set("y", text_y)
                .set("fill", controller_color_text.clone())
                .set("font-size", 20)
                .set("font-weight", "bold")
                .set("text-anchor", "middle")
                .set("font-style", "normal")
                .set("transform", format!("rotate(90, {}, {})", text_x, text_y)),
        );

        let resource_width = geometry_map["resource_width"];
        let resource_height = geometry_map["resource_height"];
        let resource_color_fill = color_map["resource_color_fill"].clone();
        let resource_color_border = color_map["resource_color_border"].clone();
        let resource_color_text = color_map["resource_color_text"].clone();
        let resource_x = x + offset + 10 + 100;
        let mut resource_y = y + offset + 50 + (16 * resource_height);
        let mut port_num = 0;
        for rs in resources {
            let resource = rs.as_object().unwrap();
            let resource_size = resource["size"].as_i64().unwrap();
            cell_group = cell_group.add(
                Rectangle::new()
                    .set("x", resource_x + 11)
                    .set("y", resource_y - resource_height * resource_size + 1)
                    .set("width", resource_width - 12)
                    .set("height", resource_height * resource_size - 2)
                    .set("fill", resource_color_fill.clone())
                    .set("stroke", resource_color_border.clone())
                    .set("stroke-width", 2),
            );

            // add port numbers to the slot
            for i in 1..(resource_size + 1) {
                let resource_x0 = resource_x + 11;
                let resource_x1 = resource_x + 11 + resource_width - 12;
                let resource_y0 = resource_y - resource_height * i + 1;
                let resource_y1 = resource_y0 + 8;
                let corner_top_left = (resource_x0 + 5, resource_y0 + 10, port_num + 3);
                let corner_btm_left = (resource_x0 + 5, resource_y1 + 10, port_num + 1);
                let corner_top_right = (resource_x1 - 5, resource_y0 + 10, port_num + 2);
                let corner_btm_right = (resource_x1 - 5, resource_y1 + 10, port_num);
                let corners = [
                    corner_top_left,
                    corner_btm_left,
                    corner_top_right,
                    corner_btm_right,
                ];
                for corner in corners.iter() {
                    cell_group = cell_group.add(
                        Text::new(corner.2.to_string())
                            .set("x", corner.0)
                            .set("y", corner.1)
                            .set("fill", resource_color_text.clone())
                            .set("font-size", 8)
                            .set("text-anchor", "middle"),
                    );
                }
                port_num += 4;
            }

            // add dotted line if the resource is bigger than 1 slot
            if resource_size > 1 {
                for i in 1..resource_size {
                    let line_x = resource_x + 11;
                    let line_y = resource_y - resource_height * (resource_size - i) + 1;
                    cell_group = cell_group.add(
                        Line::new()
                            .set("x1", line_x)
                            .set("y1", line_y)
                            .set("x2", line_x + resource_width - 12)
                            .set("y2", line_y)
                            .set("stroke", resource_color_border.clone())
                            .set("stroke-width", 1)
                            .set("stroke-dasharray", "5 5"),
                    );
                }
            }
            // add text rotate 90 degree
            let text_x = resource_x + 11 + resource_width / 2;
            let text_y = resource_y - resource_height * resource_size
                + 5
                + (resource_height * resource_size) / 2;
            cell_group = cell_group.add(
                Text::new(resource["name"].as_str().unwrap())
                    .set("x", text_x)
                    .set("y", text_y)
                    .set("fill", resource_color_text.clone())
                    .set("font-size", 16)
                    .set("font-weight", "bold")
                    .set("text-anchor", "middle")
                    .set("font-style", "normal"),
            );
            resource_y -= resource_height * resource_size;
        }

        document = document.add(cell_group);
    }

    document
}
